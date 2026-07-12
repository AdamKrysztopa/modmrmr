"""Recovery scorecard: how well a selected feature set matches a ``GroundTruth``.

Scores a selection against the informative / codependent / noise roles carried by
``GroundTruth``: precision/recall/f1 over the truly-informative set, plus two rate
diagnostics — how much of the pick was a redundant duplicate from an
already-represented codependent group, and how much was pure noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import average_precision_score, roc_auc_score

from mechanism.ground_truth import GroundTruth


@dataclass(frozen=True)
class RecoveryScore:
    precision: float
    recall: float
    f1: float
    redundancy_rate: float
    noise_rate: float


@dataclass(frozen=True)
class RankingScore:
    average_precision: float
    roc_auc: float


def recovery(selected_idx: list[int], gt: GroundTruth) -> RecoveryScore:
    """Score ``selected_idx`` against ``gt``.

    ``precision``/``recall``/``f1`` are computed against ``gt.informative`` only
    (the truly-informative set, not ``relevant_columns``). ``redundancy_rate``
    walks ``selected_idx`` in order: the first pick from a given codependent group
    is its representative (not redundant); every later pick from that same group
    counts as redundant. ``noise_rate`` is the fraction of the pick in ``gt.noise``.
    All divisions are guarded — an empty ``selected_idx`` yields an all-zero score.
    """
    n_selected = len(selected_idx)
    if n_selected == 0:
        return RecoveryScore(precision=0.0, recall=0.0, f1=0.0, redundancy_rate=0.0, noise_rate=0.0)

    informative = set(gt.informative)
    noise = set(gt.noise)

    n_hits = sum(1 for idx in selected_idx if idx in informative)

    precision = n_hits / n_selected
    recall = n_hits / len(informative) if informative else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    seen_groups: set[int] = set()
    n_redundant = 0
    for idx in selected_idx:
        group = gt.codependent_group_of(idx)
        if group is None:
            continue
        if group in seen_groups:
            n_redundant += 1
        else:
            seen_groups.add(group)
    redundancy_rate = n_redundant / n_selected

    n_noise = sum(1 for idx in selected_idx if idx in noise)
    noise_rate = n_noise / n_selected

    return RecoveryScore(
        precision=precision,
        recall=recall,
        f1=f1,
        redundancy_rate=redundancy_rate,
        noise_rate=noise_rate,
    )


def ranking_scores(selection_order: list[int], gt: GroundTruth) -> RankingScore:
    """Score the full pick ``selection_order`` against ``gt`` as a ranking problem.

    Induces a descending score over ALL ``gt.n_features`` columns: a selected
    feature at pick-rank ``r`` (0-based, best first) gets score ``n_features - r``;
    a feature never selected gets ``-1`` (below every selected score). The binary
    label is 1 iff the column is in ``gt.informative`` (informative only, not
    codependent). Reports ``average_precision`` and ``roc_auc`` over all columns.
    Guarded: if ``gt.informative`` is empty, or every column is informative (no
    negatives), the metric is undefined and ``RankingScore(0.0, 0.0)`` is returned.
    ``selection_order`` must hold distinct, in-range column indices (an mRMR
    selector never re-picks); a duplicate or out-of-range index raises ``ValueError``.
    """
    n = gt.n_features
    informative = set(gt.informative)
    if not informative or len(informative) == n:
        return RankingScore(average_precision=0.0, roc_auc=0.0)

    if len(set(selection_order)) != len(selection_order):
        raise ValueError(f"selection_order must be distinct indices, got {selection_order}")
    if any(idx < 0 or idx >= n for idx in selection_order):
        raise ValueError(f"selection_order indices must be in range(0, {n}), got {selection_order}")

    y_score = [-1.0] * n
    for rank, idx in enumerate(selection_order):
        y_score[idx] = float(n - rank)
    y_true = [1 if idx in informative else 0 for idx in range(n)]

    return RankingScore(
        average_precision=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
    )
