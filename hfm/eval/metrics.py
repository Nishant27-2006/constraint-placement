"""Evaluation metrics for probabilistic grid-scenario generation.

Three families, all reported in the paper:

1. *Proper scoring rules* against the realised day -- energy score, CRPS,
   variogram score (Gneiting & Raftery 2007; Scheuerer & Hamill 2015).  These
   are the right primary metrics: they are minimised only by the true
   predictive distribution, so they cannot be gamed by a model that is merely
   sharp or merely calibrated.
2. *Distributional distances* between the pooled generated and held-out
   populations -- sliced Wasserstein-1, MMD, per-dimension W1.
3. *Structure diagnostics* the power-systems literature relies on --
   autocorrelation, ramp-rate distribution, power spectral density, and
   rank-histogram calibration.

All estimators here are the standard *unbiased* sample versions where one
exists, and every function takes ``ens`` of shape ``(N, M, T, D)`` (N test days,
M ensemble members) with ``obs`` of shape ``(N, T, D)``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "energy_score", "crps_ensemble", "variogram_score", "sliced_w1", "mmd_rbf",
    "marginal_w1", "acf_error", "ramp_w1", "psd_error", "rank_histogram",
    "interval_coverage", "all_metrics",
]


def _flat(a):
    return a.reshape(*a.shape[:-2], -1)


def energy_score(ens: np.ndarray, obs: np.ndarray, max_pairs: int = 100,
                 seed: int = 0) -> float:
    """Unbiased energy score, averaged over test cases.  Lower is better.

    Computed one test case at a time: the naive vectorised form would
    materialise an ``(N, M, M, P)`` tensor (tens of gigabytes at benchmark
    scale), whereas the pairwise block for a single case is only ``(M, M)``.
    """
    N, M = ens.shape[0], ens.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.choice(M, size=max_pairs, replace=False) if M > max_pairs else np.arange(M)
    m = idx.size
    out = np.empty(N)
    for i in range(N):
        Ei = ens[i].reshape(M, -1).astype(np.float64)
        Yi = obs[i].reshape(-1).astype(np.float64)
        t1 = np.linalg.norm(Ei - Yi, axis=-1).mean()
        S = Ei[idx]
        sq = (S * S).sum(1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (S @ S.T), 0.0)
        t2 = np.sqrt(d2).sum() / (2.0 * m * (m - 1))
        out[i] = t1 - t2
    return float(out.mean())


def crps_ensemble(ens: np.ndarray, obs: np.ndarray, chunk: int = 32) -> float:
    """Mean univariate CRPS over all (time, channel) coordinates.

    Uses the order-statistic identity
    ``CRPS = (1/M) sum|x_i - y| - 1/(2 M^2) sum_ij |x_i - x_j|`` evaluated in
    O(M log M) by sorting, and streams over test cases so that peak memory is
    set by ``chunk`` rather than by the full ensemble.
    """
    N, M = ens.shape[0], ens.shape[1]
    i = np.arange(1, M + 1, dtype=np.float64).reshape(1, M, 1, 1)
    coef = 2.0 * (2 * i - M - 1)
    acc, cnt = 0.0, 0
    for a in range(0, N, chunk):
        E = np.sort(ens[a:a + chunk].astype(np.float64, copy=False), axis=1)
        Y = obs[a:a + chunk].astype(np.float64, copy=False)[:, None]
        t1 = np.abs(E - Y).mean(axis=1)
        t2 = (coef * E).sum(axis=1) / (2.0 * M * M)
        acc += float((t1 - t2).sum()); cnt += t1.size
    return acc / max(1, cnt)


def variogram_score(ens: np.ndarray, obs: np.ndarray, p: float = 0.5,
                    max_dims: int = 64, seed: int = 0) -> float:
    """Variogram score of order ``p`` (Scheuerer & Hamill, MWR 143:1321-1334, 2015).

    Sensitive to the *dependence* structure, which the energy score probes only
    weakly -- the failure mode of a model that gets marginals right and
    correlations wrong.  Looped over test cases for the same memory reason as
    :func:`energy_score`.
    """
    rng = np.random.default_rng(seed)
    N = ens.shape[0]
    P = ens.shape[2] * ens.shape[3]
    dims = rng.choice(P, size=min(P, max_dims), replace=False)
    tot = 0.0
    for i in range(N):
        Ei = ens[i].reshape(ens.shape[1], -1)[:, dims].astype(np.float64)
        Yi = obs[i].reshape(-1)[dims].astype(np.float64)
        vy = np.abs(Yi[:, None] - Yi[None, :]) ** p
        ve = (np.abs(Ei[:, :, None] - Ei[:, None, :]) ** p).mean(axis=0)
        tot += ((vy - ve) ** 2).sum()
    return float(tot / N)


def sliced_w1(a: np.ndarray, b: np.ndarray, n_proj: int = 512, seed: int = 0) -> float:
    """Sliced Wasserstein-1 between two point clouds of shape (n, P)."""
    rng = np.random.default_rng(seed)
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    P = a.shape[1]
    V = rng.normal(size=(P, n_proj))
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    pa = np.sort(a @ V, axis=0)
    pb = np.sort(b @ V, axis=0)
    n = min(pa.shape[0], pb.shape[0])
    qa = pa[np.linspace(0, pa.shape[0] - 1, n).astype(int)]
    qb = pb[np.linspace(0, pb.shape[0] - 1, n).astype(int)]
    return float(np.abs(qa - qb).mean())


def mmd_rbf(a: np.ndarray, b: np.ndarray, n_max: int = 1000, seed: int = 0) -> float:
    """Unbiased MMD^2 with an RBF kernel and the median-distance bandwidth."""
    rng = np.random.default_rng(seed)
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    if a.shape[0] > n_max:
        a = a[rng.choice(a.shape[0], n_max, replace=False)]
    if b.shape[0] > n_max:
        b = b[rng.choice(b.shape[0], n_max, replace=False)]
    Z = np.concatenate([a, b], 0)
    # ||z_i - z_j||^2 via the inner-product identity; forming the (n, n, P)
    # difference tensor directly would need tens of gigabytes at benchmark width.
    sq = (Z * Z).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    K = np.exp(-d2 / (med + 1e-12))
    n, m = a.shape[0], b.shape[0]
    Kaa, Kbb, Kab = K[:n, :n], K[n:, n:], K[:n, n:]
    mmd = ((Kaa.sum() - np.trace(Kaa)) / (n * (n - 1))
           + (Kbb.sum() - np.trace(Kbb)) / (m * (m - 1))
           - 2 * Kab.mean())
    return float(mmd)


def marginal_w1(gen: np.ndarray, real: np.ndarray) -> float:
    """Mean per-coordinate Wasserstein-1 (exact in 1D via sorted quantiles)."""
    g = gen.reshape(gen.shape[0], -1).astype(np.float64)
    r = real.reshape(real.shape[0], -1).astype(np.float64)
    n = min(g.shape[0], r.shape[0])
    gs = np.sort(g, 0)[np.linspace(0, g.shape[0] - 1, n).astype(int)]
    rs = np.sort(r, 0)[np.linspace(0, r.shape[0] - 1, n).astype(int)]
    return float(np.abs(gs - rs).mean())


def acf_error(gen: np.ndarray, real: np.ndarray, max_lag: int = 12) -> float:
    """Mean absolute error of the intra-day autocorrelation function."""
    def acf(x):
        x = x - x.mean(axis=1, keepdims=True)
        v = (x ** 2).mean(axis=1) + 1e-12
        out = []
        for L in range(1, max_lag + 1):
            out.append((x[:, L:] * x[:, :-L]).mean(axis=1) / v)
        return np.stack(out, 1).mean(axis=(0, 2))
    return float(np.abs(acf(gen.astype(np.float64)) - acf(real.astype(np.float64))).mean())


def ramp_w1(gen: np.ndarray, real: np.ndarray) -> float:
    """W1 between the distributions of hour-to-hour increments (ramp rates)."""
    return marginal_w1(np.diff(gen, axis=1), np.diff(real, axis=1))


def psd_error(gen: np.ndarray, real: np.ndarray) -> float:
    """Mean absolute log-power-spectral-density error over intra-day frequencies."""
    def psd(x):
        F = np.fft.rfft(x.astype(np.float64) - x.mean(axis=1, keepdims=True), axis=1)
        return np.log(np.abs(F) ** 2 + 1e-8).mean(axis=(0, 2))
    return float(np.abs(psd(gen) - psd(real)).mean())


def rank_histogram(ens: np.ndarray, obs: np.ndarray, n_bins: int = 11):
    """Rank histogram and its reliability index ``RI = sum |f_k - 1/K|``.

    ``RI = 0`` is perfect calibration; U-shaped histograms mean under-dispersion.
    """
    M, N = ens.shape[1], ens.shape[0]
    hist = np.zeros(n_bins, dtype=np.int64)
    for a in range(0, N, 32):
        r = (ens[a:a + 32] < obs[a:a + 32][:, None]).sum(axis=1).reshape(-1)
        h, _ = np.histogram(r, bins=n_bins, range=(0, M))
        hist += h
    f = hist / max(1, hist.sum())
    return f, float(np.abs(f - 1.0 / n_bins).sum())


def interval_coverage(ens: np.ndarray, obs: np.ndarray, level: float = 0.9):
    """Empirical coverage and mean width of the central ``level`` interval."""
    N = ens.shape[0]
    ncov, nwid, cnt = 0.0, 0.0, 0
    for a in range(0, N, 32):
        e = ens[a:a + 32]
        lo = np.quantile(e, (1 - level) / 2, axis=1)
        hi = np.quantile(e, 1 - (1 - level) / 2, axis=1)
        o = obs[a:a + 32]
        ncov += float(((o >= lo) & (o <= hi)).sum())
        nwid += float((hi - lo).sum()); cnt += o.size
    return ncov / cnt, nwid / cnt


def all_metrics(ens: np.ndarray, obs: np.ndarray, pooled_real: np.ndarray | None = None,
                pool_cap: int = 20000, seed: int = 0) -> dict:
    """Full metric suite.  ``ens``: (N, M, T, D); ``obs``: (N, T, D).

    The pooled population metrics are computed on a capped random subsample
    (``pool_cap``) drawn with a fixed seed, so they stay comparable across
    methods while keeping peak memory bounded.
    """
    rng = np.random.default_rng(seed)
    gen_pool = ens.reshape(-1, *ens.shape[2:])
    if gen_pool.shape[0] > pool_cap:
        gen_pool = gen_pool[rng.choice(gen_pool.shape[0], pool_cap, replace=False)]
    real_pool = obs if pooled_real is None else pooled_real
    f90, ri = rank_histogram(ens, obs)
    cov90, w90 = interval_coverage(ens, obs, 0.9)
    cov50, w50 = interval_coverage(ens, obs, 0.5)
    return {
        "energy_score": energy_score(ens, obs),
        "crps": crps_ensemble(ens, obs),
        "variogram_score": variogram_score(ens, obs),
        "sliced_w1": sliced_w1(gen_pool, real_pool),
        "mmd2": mmd_rbf(gen_pool, real_pool),
        "marginal_w1": marginal_w1(gen_pool, real_pool),
        "acf_err": acf_error(gen_pool, real_pool),
        "ramp_w1": ramp_w1(gen_pool, real_pool),
        "psd_err": psd_error(gen_pool, real_pool),
        "reliability_index": ri,
        "cov90": cov90, "width90": w90, "cov50": cov50, "width50": w50,
    }
