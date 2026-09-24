# Proofs, and the objection that survives them

Setting throughout. Scenarios live in $\mathcal X=\mathbb R^{T\times D}$.
$A\in\mathbb R^{m\times n}$, $\mathcal C=\{x: Ax=b\}\neq\emptyset$,
$V=\ker A$, $\Pi_V=I-A^{+}A$ the orthogonal projector onto $V$, and
$\Pi_{\mathcal C}(x)=x-A^{+}(Ax-b)$ the orthogonal projection onto $\mathcal C$.
Data law $q$ is supported on $\mathcal C$; base law $p_0=\Pi_{\mathcal C\#}\mathcal N(0,I)$
is too. Couple $x_0\sim p_0 \perp x_1\sim q$, set $x_t=(1-t)x_0+tx_1$, target
$u=x_1-x_0$, and let $p_t=\mathrm{Law}(x_t)$.

---

## Theorem 1 (exact, free, bias-free)

**(i) Path feasibility.** $x_t\in\mathcal C$ for all $t\in[0,1]$ and $u\in V$ a.s.

*Proof.* $\mathcal C$ is convex and $x_0,x_1\in\mathcal C$, so the convex
combination $x_t\in\mathcal C$. And $Au=Ax_1-Ax_0=b-b=0$. $\square$

**(ii) No bias.** The flow-matching optimum
$v^\star(x,t)=\mathbb E[u\mid x_t=x]$ lies in $V$ for $p_t$-a.e. $x$.

*Proof.* $V$ is a closed linear subspace and $u\in V$ a.s. by (i). Conditional
expectation of an a.s.-$V$-valued integrable random vector lies in $V$: for any
$w\perp V$, $\langle \mathbb E[u\mid x_t],w\rangle=\mathbb E[\langle u,w\rangle\mid x_t]=0$. $\square$

Consequence: restricting the hypothesis class to $V$-valued fields excludes no
minimiser. The restriction is **bias-free**, not a relaxation.

**(iii) Monotone improvement.** For every measurable $v$ with
$\mathbb E\|v(x_t,t)\|^2<\infty$,
$$\mathcal L(\Pi_V v)=\mathcal L(v)-\mathbb E\big\|(I-\Pi_V)v(x_t,t)\big\|^2\le \mathcal L(v),$$
with equality iff $v$ is $V$-valued $p_t$-a.e.

*Proof.* Pointwise, write $v=\Pi_V v+(I-\Pi_V)v$. Since $u\in V$ and
$(I-\Pi_V)v\perp V$, the cross term vanishes:
$\|v-u\|^2=\|\Pi_V v-u\|^2+\|(I-\Pi_V)v\|^2$. Take expectations over
$(t,x_0,x_1)$. $\square$

**(iv) Exactness under any Runge–Kutta scheme.** If $x\in\mathcal C$ and an
integrator forms $x^{+}=x+h\sum_i b_i k_i$ where each stage $k_i$ is an
evaluation of a $V$-valued field, then $x^{+}\in\mathcal C$ — for any step
sizes, any stage count, any order.

*Proof.* $Ax^{+}=Ax+h\sum_i b_i(Ak_i)=Ax=b$ since $Ak_i=0$. Induct. $\square$

No tolerance parameter, no inner solve, no dependence on step count. This is why
the measured residual is flat at $10^{-14}$ after 50 Euler steps
(`docs/VERIFICATION_LOG.md` §1) and identical across four solvers (§2).

---

## Proposition 2 (drift control on a nonlinear manifold)

Let $\mathcal M=\{x:g(x)=0\}$ with $g\in C^2$, $J=\nabla g$ of full row rank with
bounded $J^{+}$ near the trajectory, and let $v$ be $C^1$ and tangential
($Jv=0$ on $\mathcal M$). An order-$p$ Runge–Kutta step of size $h$ gives
$\|g(x_{k+1})\|=\|g(x_k)\|+O(h^{p+1})$, so after $K=1/h$ steps the invariant has
drifted by $O(h^{p})$. One Gauss–Newton retraction $R(x)=x-J^{+}g(x)$ contracts
quadratically, $\|g(R(x))\|=O(\|g(x)\|^2)$, so retracting every step bounds the
terminal residual at $O(h^{2(p+1)})$.

*Proof sketch.* Taylor-expand $g$ along the step; the $O(h)$ term vanishes by
tangency and the leading surviving term is the local truncation error at
$O(h^{p+1})$. Accumulation over $1/h$ steps gives $O(h^p)$. The retraction is a
Newton step on the least-squares system $g=0$; standard Newton–Kantorovich gives
quadratic local contraction. $\square$

This is the classical projection-method result (Hairer–Lubich–Wanner, Ch. IV.4)
specialised to a learned tangential field. **We verify the rates numerically
rather than resting on the constants**, because the constants depend on the
learned $v_\theta$, not only on $g$. If the measured exponents disagree, the
proposition is reported as not empirically confirmed.

---

## Proposition 3 — the objection that survives, and why we state it ourselves

> **Post-hoc projection can never increase Wasserstein distance to the data.**
> Let $p_\theta$ be any generated law and $q$ the data law supported on
> $\mathcal C$. Then for every $r\ge 1$,
> $$W_r\big(\Pi_{\mathcal C\#}p_\theta,\;q\big)\;\le\;W_r\big(p_\theta,\;q\big).$$

*Proof.* Orthogonal projection onto a non-empty closed convex set is
1-Lipschitz, and $\Pi_{\mathcal C\#}q=q$ because $q$ is supported on
$\mathcal C$. For any coupling $\gamma$ of $(p_\theta,q)$, the pushforward
$(\Pi_{\mathcal C}\times\Pi_{\mathcal C})_\#\gamma$ couples
$\Pi_{\mathcal C\#}p_\theta$ and $q$, and its cost is no larger since
$\|\Pi x-\Pi y\|\le\|x-y\|$ pointwise. Take the infimum. $\square$

**Why this matters, and why it is in the paper rather than buried.** A referee
will raise it, and it is correct: geometrically, simply projecting the samples at
the end is never harmful. So Theorem 1 does **not** establish that train-time
projection beats post-hoc projection. Theorem 1(iii) compares $\Pi_V v$ to $v$
for a *fixed* $v$; it says nothing about how the two *trained* fields
$p_\theta^{\text{proj}}$ and $\Pi_{\mathcal C\#}p_\theta^{\text{unconstrained}}$
compare, because training changes $\theta$.

The honest position, which is the paper's actual thesis:

* **Geometry gives no ordering.** Proposition 3 rules out an a-priori argument
  in either direction.
* **The mechanism we hypothesise is optimisation, not geometry.** A projected
  network never needs to represent the $\operatorname{rank}(A)$ directions whose
  answer is known in closed form — 197 of 304 channel-directions per hour in
  Suite B. Capacity and gradient signal are reallocated to the 107 directions
  that carry actual uncertainty.
* **Therefore it is an empirical question**, and it is the question the
  benchmark is built to answer (claim C3). We pre-committed in
  `docs/CLAIMS_AND_EVIDENCE.md` to reporting "no difference" as the result if
  the seeds say so.

Proposition 3 also explains why `FM+posthoc` is a *strong* baseline and not a
straw man, and it is the reason we run it at all.

---

## What is deliberately **not** claimed

1. **Not novel: hard constraints in flow matching.** Utkarsh et al., *Physics-
   Constrained Flow Matching* (NeurIPS 2025, arXiv:2506.04171) achieves exact
   satisfaction of arbitrary nonlinear constraints zero-shot on pretrained flow
   models. We reimplement it as a baseline.
2. **Not novel: adaptive-step generative sampling.** `Gotta Go Fast`
   (arXiv:2105.14080) and standard `dopri5` predate any adaptive schedule here.
   We benchmark against `dopri5` at matched tolerance and claim nothing else.
3. **Not a Hamiltonian method.** There is no symplectic form on this state space.
   The tangential flow conserves the constraint functions as first integrals,
   which is weaker. "Hamiltonian Generative Flows" is already taken, with a
   different meaning, by Holderrieth, Xu & Jaakkola (NeurIPS 2024).
4. **Exact means exact in real arithmetic.** In `float32` the residual is
   round-off-limited (`7.4e-06`); in `float64` it returns to `2.4e-14`. Both are
   reported.
