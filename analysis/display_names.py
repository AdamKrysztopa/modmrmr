"""Shared display-name mapping for paper artifacts (tables + figures).

The design-space operator token ``multiplicative`` is the raw spec/CSV token
for the paper's novel operator, which is *called* "the gate" in prose. Released
CSV artifacts (e.g. ``results/factorial_leaderboard.csv``) keep the raw token —
only generated display surfaces (table cells, spec strings, axis labels,
legends) route through :func:`display_token` / :func:`display_spec` so the
token never needs rewriting in two places.
"""

from __future__ import annotations

# Raw token -> paper display name. Applied to individual operator/aggregation
# tokens and to each ``|``-delimited segment of a spec label.
DISPLAY_NAMES: dict[str, str] = {
    "multiplicative": "gate",
    "reg_quotient": "reg. quotient",
}


def display_token(token: str) -> str:
    """Map a single raw token (operator name, variant key, ...) to its display form."""
    return DISPLAY_NAMES.get(token, token)


def display_spec(spec: str) -> str:
    """Map a ``relevance|redundancy|operator|aggregation`` spec label to display form.

    Applies :func:`display_token` to each ``|``-delimited segment, so
    ``"dcor|pearson_abs|multiplicative|mean"`` renders as
    ``"dcor|pearson_abs|gate|mean"``.
    """
    return "|".join(display_token(part) for part in spec.split("|"))
