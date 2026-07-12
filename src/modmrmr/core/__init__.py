"""General, domain-free ModMRMR core."""

from __future__ import annotations

from modmrmr.core.estimator import MRMR, ModMRMR, MRMRSelector, pearson_corr

__all__ = ["MRMR", "MRMRSelector", "ModMRMR", "pearson_corr"]
