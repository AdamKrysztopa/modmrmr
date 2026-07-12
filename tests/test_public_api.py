import modmrmr


def test_top_level_exports_present() -> None:
    for name in [
        "MRMR",
        "ModMRMR",
        "MRMRSelector",
        "run_tmrmr",
        "register_scorer",
        "get_scorer",
        "list_scorers",
        "as_importance_function",
        "as_penalty_matrix",
        "LagAwareModMRMRConfig",
        "PairwiseScorerSpec",
    ]:
        assert hasattr(modmrmr, name), name
