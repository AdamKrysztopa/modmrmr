"""Lag-aware ModMRMR regression tests against frozen characterization fixtures.

Compares the ported ModMRMR engine output against the frozen fixtures under
``tests/fixtures/expected/``, over the ModMRMR-owned fields only. Their purpose,
oracle limitations, and update policy are documented in
``tests/fixtures/expected/README.md``. The
downstream forecast-prep-contract export fields (``contract`` / ``lag_table``
/ ``markdown_length``) are intentionally out of scope and skipped by the
comparison (see ``modmrmr.diagnostics.regression``).
"""

from __future__ import annotations

import pytest

from modmrmr.diagnostics.regression import (
    _OUT_OF_SCOPE_FIXTURE_KEYS,
    LAG_AWARE_MOD_MRMR_FIXTURE_CASES,
    build_scenario,
    compare_to_fixture,
    load_fixture,
)


class TestLagAwareModMRMRRegressionMatchesFrozen:
    """Rebuilt outputs must match the frozen characterization (in-scope keys)."""

    @pytest.mark.parametrize("case_name", LAG_AWARE_MOD_MRMR_FIXTURE_CASES)
    def test_scenario_matches_frozen_expected(self, case_name: str) -> None:
        """Each scenario reproduces every ModMRMR-owned fixture field."""
        result = build_scenario(case_name)
        fixture = load_fixture(case_name)
        mismatches = compare_to_fixture(result, fixture)
        assert mismatches == [], f"[{case_name}] engine-field mismatch(es):\n" + "\n".join(
            f"- {line}" for line in mismatches
        )

    def test_all_expected_files_present(self) -> None:
        """Every fixture case must have a corresponding expected JSON file."""
        for case_name in LAG_AWARE_MOD_MRMR_FIXTURE_CASES:
            # load_fixture raises FileNotFoundError if the characterization is absent.
            fixture = load_fixture(case_name)
            assert fixture["case_name"] == case_name


class TestLagAwareModMRMRRegressionDriftDetection:
    """Verification must fail when any rebuilt engine-owned field drifts."""

    def test_mutated_selected_lag_flags_drift(self) -> None:
        """Mutating a selected feature's lag should trigger a comparison failure."""
        case_name = "known_synthetic_lag_driver"
        fixture = load_fixture(case_name)
        result = build_scenario(case_name)

        # Sanity: the pristine result matches.
        assert compare_to_fixture(result, fixture) == []

        # Corrupt an engine-owned field and confirm the mismatch is reported.
        assert isinstance(result["selected"], list) and result["selected"]
        result["selected"][0]["lag"] = 99  # type: ignore[index]
        mismatches = compare_to_fixture(result, fixture)
        assert mismatches, "Expected a mismatch after mutating a selected lag"
        assert any(case_name in line for line in mismatches)

    def test_out_of_scope_keys_are_ignored(self) -> None:
        """Differences confined to out-of-scope keys must not be reported."""
        case_name = "duplicate_sensor_suppression"
        fixture = load_fixture(case_name)
        result = build_scenario(case_name)

        # Injecting a bogus out-of-scope key into the fixture side must be ignored.
        for key in _OUT_OF_SCOPE_FIXTURE_KEYS:
            fixture[key] = {"anything": "downstream"}
        assert compare_to_fixture(result, fixture) == []


class TestNotesCarryNoAttribution:
    """Runtime notes describe the method, not who proposed it."""

    def test_no_fixture_note_names_an_author(self) -> None:
        """Author attribution must not be load-bearing in the fixtures."""
        for case_name in LAG_AWARE_MOD_MRMR_FIXTURE_CASES:
            fixture = load_fixture(case_name)
            for note in fixture.get("notes", []):
                assert "proposed by" not in note, (
                    f"[{case_name}] fixture note attributes authorship: {note!r}"
                )

    def test_engine_emits_no_attribution_note(self) -> None:
        """The engine itself must not emit an attribution note."""
        result = build_scenario("duplicate_sensor_suppression")
        for note in result["notes"]:
            assert "proposed by" not in note
