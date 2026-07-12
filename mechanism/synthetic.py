"""Seeded synthetic generators for the mechanism suite.

Each generator returns ``(X, y, GroundTruth)``. Every dataset is a realistic mix of
INFORMATIVE features (the mechanism), CO-DEPENDENT features (dependent on an
informative one — linearly and/or NONLINEARLY, since redundancy is not only
collinearity), and pure NOISE. Columns are shuffled; the GroundTruth records the
true positions. All randomness flows from ``np.random.default_rng(seed)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mechanism.ground_truth import GroundTruth


def _assemble(rng, columns, roles, y, dependence):
    order = rng.permutation(len(columns))
    shuffled = [columns[i] for i in order]
    X = pd.DataFrame({f"f{pos}": col for pos, col in enumerate(shuffled)})
    informative: list[int] = []
    groups: dict[int, list[int]] = {}
    noise: list[int] = []
    for new_pos, old_idx in enumerate(order):
        role = roles[old_idx]
        if role == "info":
            informative.append(new_pos)
        elif role == "noise":
            noise.append(new_pos)
        else:  # ("codep", group_id)
            groups.setdefault(role[1], []).append(new_pos)
    codependent = tuple(tuple(sorted(v)) for _, v in sorted(groups.items()))
    gt = GroundTruth(
        informative=tuple(sorted(informative)),
        codependent=codependent,
        noise=tuple(sorted(noise)),
        dependence=dependence,
        n_features=len(columns),
    )
    return X, pd.Series(np.asarray(y), name="target"), gt


def _noise_cols(rng, n, k):
    return [rng.normal(size=n) for _ in range(k)]


def make_parabola(seed=0, n=800):
    """Reg: y = x0**2 + eps. Pearson(x0,y)~0, MI high. Codependent: linear twin
    (x0 + small noise) and nonlinear twin (x0**2 + small noise); + noise cols."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    y = x0**2 + 0.1 * rng.normal(size=n)
    linear_twin = x0 + 0.05 * rng.normal(size=n)
    nonlinear_twin = x0**2 + 0.05 * rng.normal(size=n)
    cols = [x0, linear_twin, nonlinear_twin, *_noise_cols(rng, n, 5)]
    roles = ["info", ("codep", 0), ("codep", 0), "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


def make_radial(seed=0, n=800):
    """Clf: y = 1[x0^2 + x1^2 > median]. Each of x0,x1 marginally MI>0, Pearson~0."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    r2 = x0**2 + x1**2
    y = (r2 > np.median(r2)).astype(int)
    nonlinear_twin = np.abs(x0) + 0.05 * rng.normal(size=n)  # codependent with x0 nonlinearly
    cols = [x0, x1, nonlinear_twin, *_noise_cols(rng, n, 5)]
    roles = ["info", "info", ("codep", 0), "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


def make_sine(seed=0, n=800):
    """Reg: y = sin(3*x0) + eps. Codependent: nonlinear twin |x0| + noise.

    x0 is drawn Laplace(0, 1) (symmetric, heavier-tailed than Normal/Uniform) rather
    than Uniform(-pi, pi): a sin(k*x0) twin is oscillatory enough that its distance
    correlation with x0 stays well under the 0.5 axis-bites threshold regardless of
    sample size (verified analytically/empirically), so the nonlinear twin uses the
    even "fold" function |x0| instead — same "pearson-blind, dcor-visible" property
    already used successfully in make_radial/make_parabola, but with enough margin
    (dcor ~0.6, pearson <0.15) to clear the threshold robustly.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.laplace(size=n)
    y = np.sin(3.0 * x0) + 0.1 * rng.normal(size=n)
    nonlinear_twin = np.abs(x0) + 0.05 * rng.normal(size=n)
    linear_twin = x0 + 0.05 * rng.normal(size=n)
    cols = [x0, linear_twin, nonlinear_twin, *_noise_cols(rng, n, 5)]
    roles = ["info", ("codep", 0), ("codep", 0), "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


def _nonlin_redundancy(rng, n, task):
    # Laplace x0 + even "fold" twin (same as make_sine): pearson(x0,·)~0 but
    # dcor(x0,|x0|) ~0.6 (Laplace's heavier tails lift it well clear of the 0.5
    # threshold vs ~0.53-0.56 for a squared/Gaussian twin). Relevance to y stays
    # strongly linear (y is unchanged).
    x0 = rng.laplace(size=n)
    nonlinear_twin = np.abs(x0) + 0.05 * rng.normal(size=n)
    if task == "classification":
        y = (x0 + 0.3 * rng.normal(size=n) > 0).astype(int)
        dep = "mixed"
    else:
        y = 2.0 * x0 + 0.3 * rng.normal(size=n)
        dep = "mixed"
    cols = [x0, nonlinear_twin, *_noise_cols(rng, n, 5)]
    roles = ["info", ("codep", 0), "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, dep)


def make_nonlin_redundancy_clf(seed=0, n=800):
    return _nonlin_redundancy(np.random.default_rng(seed), n, "classification")


def make_nonlin_redundancy_reg(seed=0, n=800):
    return _nonlin_redundancy(np.random.default_rng(seed), n, "regression")


def _linear_control(rng, n, task):
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    linear_twin = x0 + 0.05 * rng.normal(size=n)
    if task == "classification":
        y = (x0 + x1 + 0.3 * rng.normal(size=n) > 0).astype(int)
    else:
        y = 1.5 * x0 + 1.0 * x1 + 0.2 * rng.normal(size=n)
    cols = [x0, x1, linear_twin, *_noise_cols(rng, n, 5)]
    roles = ["info", "info", ("codep", 0), "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "linear")


def make_linear_control_clf(seed=0, n=800):
    return _linear_control(np.random.default_rng(seed), n, "classification")


def make_linear_control_reg(seed=0, n=800):
    return _linear_control(np.random.default_rng(seed), n, "regression")


def make_mixed(seed=0, n=800):
    """Clf: linear informative (x0) + nonlinear informative (x1 via x1**2) +
    linear codependent twin of x0 + nonlinear codependent twin of x1 + noise."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    y = ((x0 + (x1**2 - 1.0)) + 0.3 * rng.normal(size=n) > 0).astype(int)
    lin_twin = x0 + 0.05 * rng.normal(size=n)  # codep group 0 (linear)
    nonlin_twin = x1**2 + 0.05 * rng.normal(size=n)  # codep group 1 (nonlinear)
    cols = [x0, x1, lin_twin, nonlin_twin, *_noise_cols(rng, n, 5)]
    roles = [
        "info",
        "info",
        ("codep", 0),
        ("codep", 1),
        "noise",
        "noise",
        "noise",
        "noise",
        "noise",
    ]
    return _assemble(rng, cols, roles, y, "mixed")


def _redundant_blocks(rng, n, task):
    """Redundancy trap: graded-relevance signals whose STRONG twins outrank a
    WEAK true signal on |Pearson(., y)| — a redundancy-blind (or weakly
    penalizing) operator picks the strong signal's twins over the fresh weak
    signal, hurting recall and inflating redundancy_rate. All twins are LINEAR
    so Pearson AND MI redundancy both see them: the discriminator is the
    *operator*, not the dependence measure.
    """
    x1 = rng.normal(size=n)  # STRONG, coef 3.0
    x2 = rng.normal(size=n)  # MEDIUM, coef 1.5
    x3 = rng.normal(size=n)  # WEAK, coef 0.8 — no twin
    eps = rng.normal(size=n)
    signal = 3.0 * x1 + 1.5 * x2 + 0.8 * x3 + 0.3 * eps
    if task == "classification":
        y = (signal > 0).astype(int)
    else:
        y = signal
    x1_twins = [x1 + s * rng.normal(size=n) for s in (0.05, 0.1, 0.15)]
    x2_twins = [x2 + s * rng.normal(size=n) for s in (0.05, 0.1)]
    cols = [x1, x2, x3, *x1_twins, *x2_twins, *_noise_cols(rng, n, 6)]
    roles = [
        "info",
        "info",
        "info",
        ("codep", 0),
        ("codep", 0),
        ("codep", 0),
        ("codep", 1),
        ("codep", 1),
        "noise",
        "noise",
        "noise",
        "noise",
        "noise",
        "noise",
    ]
    return _assemble(rng, cols, roles, y, "linear")


def make_redundant_blocks_clf(seed=0, n=1200):
    return _redundant_blocks(np.random.default_rng(seed), n, "classification")


def make_redundant_blocks_reg(seed=0, n=1200):
    return _redundant_blocks(np.random.default_rng(seed), n, "regression")


def _quotient_trap(rng, n, task):
    """The W->0 crowning trap: 3 moderate-coef informative signals, 2 fully
    redundant twins of x1 (W->1, tests the veto), and 8 INDEPENDENT distractor
    columns with no relation to y (W->0, the ratio/quotient operator's failure
    mode — dividing by a near-zero redundancy inflates them)."""
    x1 = rng.normal(size=n)  # informative, coef 1.5
    x2 = rng.normal(size=n)  # informative, coef 1.2
    x3 = rng.normal(size=n)  # informative, coef 1.0
    s = 1.5 * x1 + 1.2 * x2 + 1.0 * x3
    if task == "classification":
        y = (s > 0).astype(int)
    else:
        y = s + 0.3 * rng.normal(size=n)
    twins = [x1 + 0.03 * rng.normal(size=n) for _ in range(2)]
    cols = [x1, x2, x3, *twins, *_noise_cols(rng, n, 8)]
    roles = ["info", "info", "info", ("codep", 0), ("codep", 0), *(["noise"] * 8)]
    return _assemble(rng, cols, roles, y, "linear")


def make_quotient_trap_reg(seed=0, n=1200):
    return _quotient_trap(np.random.default_rng(seed), n, "regression")


def make_quotient_trap_clf(seed=0, n=1200):
    return _quotient_trap(np.random.default_rng(seed), n, "classification")


def make_xor(seed=0, n=800):
    """Clf interaction case: y = sign(x0) XOR sign(x1). Each feature ALONE carries
    ~0 mutual information with y; only the pair is informative — so pairwise
    relevance (linear OR MI) fails, motivating conditional methods (JMI/CMIM)."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    y = ((x0 > 0) ^ (x1 > 0)).astype(int)
    cols = [x0, x1, *_noise_cols(rng, n, 6)]
    roles = ["info", "info", "noise", "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


# --------------------------------------------------------------------------
# Extended nonlinear golden suite — 11 mechanistically DISTINCT generators.
# Each is a single task head with dependence="nonlinear": the relevance signal
# is (near-)invisible to Pearson but recoverable by MI / distance correlation.
# They exist to give the within-class (nonlinear) Friedman test real power, so
# no two share a mechanism (no clf/reg twins).
# --------------------------------------------------------------------------


def make_checkerboard(seed=0, n=800):
    """Clf: y = parity of floor(x0)+floor(x1) over U(-2,2) unit tiles — a
    fine-grained checkerboard (period 1, harder than xor's single sign split).
    Each feature alone is MI~0 and Pearson~0; only the pair, at tile resolution,
    is informative."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.uniform(-2.0, 2.0, size=n), rng.uniform(-2.0, 2.0, size=n)
    y = ((np.floor(x0) + np.floor(x1)) % 2).astype(int)
    cols = [x0, x1, *_noise_cols(rng, n, 6)]
    roles = ["info", "info", "noise", "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


def make_annulus(seed=0, n=800):
    """Clf: y = 1 iff the radius of (x0,x1) falls inside a ring (the inter-quartile
    band of r). Balanced ~50/50; each feature is marginally MI>0 (large |x| lands
    outside the ring) but Pearson~0."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    r = np.sqrt(x0**2 + x1**2)
    lo, hi = np.quantile(r, 0.25), np.quantile(r, 0.75)
    y = ((r > lo) & (r < hi)).astype(int)
    cols = [x0, x1, *_noise_cols(rng, n, 6)]
    roles = ["info", "info", "noise", "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


def make_multiplicative_interaction(seed=0, n=800):
    """Reg: y = x0 * x1 + eps. Informative ONLY via the product — each factor is
    zero-mean and independent so Pearson(x0,y)~0 and Pearson(x1,y)~0, yet each
    factor sets the conditional spread of y so marginal MI>0."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    y = x0 * x1 + 0.1 * rng.normal(size=n)
    cols = [x0, x1, *_noise_cols(rng, n, 6)]
    roles = ["info", "info", "noise", "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


def make_threshold_and(seed=0, n=800):
    """Clf: y = 1 iff (|x0|>t) AND (|x1|>t), a symmetric magnitude-band conjunction
    — the label fires only when BOTH features are jointly in their outer band. The
    band is even in each feature (Pearson~0), the threshold balances the classes,
    and the conjunction is the interaction that xor's parity is not."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    t = np.quantile(np.abs(np.concatenate([x0, x1])), 1.0 - np.sqrt(0.5))
    y = ((np.abs(x0) > t) & (np.abs(x1) > t)).astype(int)
    cols = [x0, x1, *_noise_cols(rng, n, 6)]
    roles = ["info", "info", "noise", "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


def make_saturation(seed=0, n=800):
    """Reg: logistic saturation of x0^2 — y = sigmoid(3*(x0**2 - 1)) + eps. The
    response is bounded and even in x0 (Pearson~0) with a clear MI signal.
    Codependent: a nonlinear fold-twin |x0|."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    y = 1.0 / (1.0 + np.exp(-3.0 * (x0**2 - 1.0))) + 0.05 * rng.normal(size=n)
    nonlinear_twin = np.abs(x0) + 0.05 * rng.normal(size=n)
    cols = [x0, nonlinear_twin, *_noise_cols(rng, n, 5)]
    roles = ["info", ("codep", 0), "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


def make_ratio(seed=0, n=800):
    """Reg: y = x0**2 / (|x1| + 0.5) + eps — a quotient mechanism where x0 sets the
    scale and x1 nonlinearly gates it. Both features enter through even functions
    (square / fold) so Pearson~0, and the ratio structure keeps MI strong."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    y = x0**2 / (np.abs(x1) + 0.5) + 0.1 * rng.normal(size=n)
    cols = [x0, x1, *_noise_cols(rng, n, 6)]
    roles = ["info", "info", "noise", "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


def make_max_of(seed=0, n=800):
    """Reg: y = max(|x0|, |x1|, |x2|) + eps — order-statistic dependence over three
    informative features. Even in each feature (Pearson~0); each carries MI>0
    because it sets y whenever it is the running maximum."""
    rng = np.random.default_rng(seed)
    x0, x1, x2 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = np.maximum(np.maximum(np.abs(x0), np.abs(x1)), np.abs(x2)) + 0.1 * rng.normal(size=n)
    cols = [x0, x1, x2, *_noise_cols(rng, n, 5)]
    roles = ["info", "info", "info", "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


def make_heteroscedastic(seed=0, n=800):
    """Reg: the informative features drive the VARIANCE of y, not its mean —
    y = (0.3 + |x0| + |x1|) * z with z~N(0,1). y is symmetric given x0 so
    Pearson(x0,y)~0, but the conditional spread encodes x0 so MI>0. Invisible to
    correlation by construction."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    z = rng.normal(size=n)
    y = (0.3 + np.abs(x0) + np.abs(x1)) * z
    cols = [x0, x1, *_noise_cols(rng, n, 6)]
    roles = ["info", "info", "noise", "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


def make_sinc_2d(seed=0, n=800):
    """Reg: y = sinc(radius) over (x0,x1) — y = sin(pi*r)/(pi*r), r=sqrt(x0^2+x1^2).
    Concentric ripples: the mean of y is a non-monotone radial function so
    Pearson~0 while MI>0."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.uniform(-3.0, 3.0, size=n), rng.uniform(-3.0, 3.0, size=n)
    r = np.sqrt(x0**2 + x1**2)
    y = np.sinc(r) + 0.05 * rng.normal(size=n)
    cols = [x0, x1, *_noise_cols(rng, n, 6)]
    roles = ["info", "info", "noise", "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


def make_polynomial_sign(seed=0, n=800):
    """Clf: y = 1 iff |x0**3 - x0| exceeds its median — the label tracks the
    MAGNITUDE of an odd cubic, giving a symmetric multi-band decision region in
    x0. Balanced and even (Pearson~0) with a strong MI signal."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    poly = x0**3 - x0
    y = (np.abs(poly) > np.median(np.abs(poly))).astype(int)
    cols = [x0, *_noise_cols(rng, n, 7)]
    roles = ["info", "noise", "noise", "noise", "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")


def make_conditional_redundancy(seed=0, n=800):
    """Clf: informative pair (x0,x1) sets a radial label y = 1[x0^2+x1^2 > median],
    plus a NONLINEAR codependent group — two fold-twins |x0| redundant with x0
    (and each other) through the label's magnitude structure. Exercises a
    non-empty codependent group under a nonlinear relevance regime."""
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    r2 = x0**2 + x1**2
    y = (r2 > np.median(r2)).astype(int)
    twin_a = np.abs(x0) + 0.05 * rng.normal(size=n)
    twin_b = np.abs(x0) + 0.05 * rng.normal(size=n)
    cols = [x0, x1, twin_a, twin_b, *_noise_cols(rng, n, 4)]
    roles = ["info", "info", ("codep", 0), ("codep", 0), "noise", "noise", "noise", "noise"]
    return _assemble(rng, cols, roles, y, "nonlinear")
