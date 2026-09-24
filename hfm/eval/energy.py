"""Honest inference-cost accounting.

The draft this work replaces reported joules per scenario from a GPU it never
ran on.  We report three quantities instead, in decreasing order of how much
weight the paper puts on them:

1. **NFE** -- velocity-network evaluations per scenario.  Exact, integer,
   hardware-independent, and the quantity the method actually changes.
2. **FLOPs** -- analytic multiply-accumulate count for those evaluations.
   Exact and reproducible on any machine (see ``hfm.models.nets.flops_per_eval``).
3. **Measured energy** -- whole-system electrical energy from the platform's
   own power telemetry, with an idle baseline subtracted.  On Apple silicon this
   reads the SMC battery gauge (``ioreg -rn AppleSmartBattery``: instantaneous
   amperage x pack voltage), which requires no elevated privileges but measures
   the *entire machine*, so it is an upper bound on the model's own consumption
   and is reported as such.  Requires the machine to be on battery; when it is
   not, the meter reports ``available = False`` and the paper falls back to
   NFE/FLOPs only rather than inventing a number.
"""

from __future__ import annotations

import re
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field

__all__ = ["PowerMeter", "measure_energy", "system_power_w"]

_INT64 = 1 << 64


def _ioreg_battery() -> dict:
    try:
        out = subprocess.run(["ioreg", "-rn", "AppleSmartBattery"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {}
    d = {}
    for key in ("InstantAmperage", "Amperage", "Voltage"):
        m = re.search(rf'"{key}"\s*=\s*(\d+)', out)
        if m:
            v = int(m.group(1))
            if v > _INT64 // 2:
                v -= _INT64
            d[key] = v
    d["external"] = '"ExternalConnected" = Yes' in out
    return d


def system_power_w():
    """Instantaneous whole-system power in watts, or ``None`` if unavailable."""
    d = _ioreg_battery()
    if not d or d.get("external", True):
        return None
    amps = d.get("InstantAmperage", d.get("Amperage"))
    volts = d.get("Voltage")
    if amps is None or volts is None:
        return None
    p = abs(amps) / 1000.0 * (volts / 1000.0)
    return p if 0.5 < p < 400.0 else None


@dataclass
class PowerMeter:
    """Background sampler of whole-system power draw."""
    interval: float = 0.25
    samples: list = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thr: threading.Thread | None = None

    def __enter__(self):
        self.samples = []
        self._stop.clear()
        self.t0 = time.perf_counter()

        def loop():
            while not self._stop.is_set():
                p = system_power_w()
                if p is not None:
                    self.samples.append((time.perf_counter(), p))
                self._stop.wait(self.interval)

        self._thr = threading.Thread(target=loop, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2.0)
        self.elapsed = time.perf_counter() - self.t0

    @property
    def available(self) -> bool:
        return len(self.samples) >= 2

    @property
    def mean_power_w(self):
        return statistics.mean(p for _, p in self.samples) if self.available else None

    @property
    def energy_j(self):
        return self.mean_power_w * self.elapsed if self.available else None


def measure_energy(fn, n_scenarios: int, idle_power_w: float | None = None,
                   warmup: int = 1, repeats: int = 3) -> dict:
    """Run ``fn`` and report per-scenario cost.

    ``idle_power_w`` should come from :func:`idle_baseline` measured immediately
    before, so that the reported figure is the *marginal* energy of generation
    rather than the machine's floor.
    """
    for _ in range(warmup):
        fn()
    lat, en = [], []
    for _ in range(repeats):
        with PowerMeter() as pm:
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
        lat.append(dt)
        if pm.available:
            p = pm.mean_power_w
            marginal = p - idle_power_w if idle_power_w is not None else p
            en.append(max(marginal, 0.0) * dt)
    out = {
        "latency_s_total": statistics.median(lat),
        "latency_ms_per_scenario": 1e3 * statistics.median(lat) / max(1, n_scenarios),
        "power_available": bool(en),
    }
    if en:
        out["energy_j_total"] = statistics.median(en)
        out["energy_j_per_scenario"] = statistics.median(en) / max(1, n_scenarios)
    return out


def idle_baseline(seconds: float = 4.0) -> float | None:
    """Mean whole-system power with no benchmark work running."""
    with PowerMeter() as pm:
        time.sleep(seconds)
    return pm.mean_power_w if pm.available else None
