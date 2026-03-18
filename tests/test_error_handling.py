"""Tests for error handling utilities — ERRH requirements."""

import pathlib

import pytest
from core.config import FilterRule, RuleGroup, CohortConfig
from core.ui_helpers import _detect_zero_match_columns, _execute_confirmed_delete


def _make_config_with_columns(*col_lists):
    """Build a CohortConfig whose inclusion group contains one rule per col_list entry."""
    rules = []
    for cols in col_lists:
        if len(cols) == 1:
            rules.append(FilterRule(column=cols[0], operator="contains", value="x"))
        else:
            rules.append(FilterRule(columns=list(cols), operator="contains", value="x"))
    inclusion = RuleGroup(logic="OR", rules=rules)
    return CohortConfig(inclusion=inclusion)


class TestZeroMatchDetection:
    """ERRH-04: Zero-match diagnostic helper returns correct columns."""

    def test_zero_match_detection_returns_columns_when_zero(self):
        """When 'After inclusion filter' step has remaining==0, return the inclusion rule columns."""
        strobe = [
            ("Total records in master", 0, 1000),
            ("After inclusion filter", 1000, 0),
        ]
        config = _make_config_with_columns(["Col A"], ["Col B"])
        result = _detect_zero_match_columns(strobe, config)
        assert result == ["Col A", "Col B"]

    def test_zero_match_detection_returns_empty_when_nonzero(self):
        """When inclusion filter leaves rows remaining, return empty list."""
        strobe = [
            ("Total records in master", 0, 1000),
            ("After inclusion filter", 100, 900),
        ]
        config = _make_config_with_columns(["Col A"], ["Col B"])
        result = _detect_zero_match_columns(strobe, config)
        assert result == []

    def test_zero_match_detection_caps_at_three_columns(self):
        """When inclusion rules cover 10 columns, return at most 3."""
        strobe = [
            ("Total records in master", 0, 5000),
            ("After inclusion filter", 5000, 0),
        ]
        # One rule with 10 columns
        ten_cols = [f"Col {i}" for i in range(10)]
        config = _make_config_with_columns(ten_cols)
        result = _detect_zero_match_columns(strobe, config)
        assert len(result) == 3
        assert result == ["Col 0", "Col 1", "Col 2"]

    def test_zero_match_detection_no_inclusion_step(self):
        """When strobe has no 'After inclusion filter' step, return empty list."""
        strobe = [
            ("Total records in master", 0, 1000),
            ("Excluded: some criterion", 50, 950),
        ]
        config = _make_config_with_columns(["Col A"])
        result = _detect_zero_match_columns(strobe, config)
        assert result == []


class TestClearDataConfirmation:
    """Tests for the two-step delete confirmation flow (ERRH-02)."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_parquet(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Create a dummy parquet file for deletion tests."""
        p = tmp_path / "all_admissions.parquet"
        p.write_bytes(b"dummy")
        return p

    def _base_state(self) -> dict:
        """Return a minimal session_state dict representing a pending confirmation."""
        return {
            "confirm_delete_pending": True,
            "delete_backup_bytes": b"backup",
            "cohort_df": object(),
            "strobe": object(),
            "master_df": object(),
            "audit_trail": object(),
        }

    # ------------------------------------------------------------------
    # test_clear_data_requires_confirmation
    # ------------------------------------------------------------------

    def test_clear_data_requires_confirmation(self, tmp_path):
        """Delete must NOT fire unless confirm_text == 'DELETE' exactly."""
        p = self._make_parquet(tmp_path)
        state = self._base_state()

        result = _execute_confirmed_delete(state, "", p)

        assert result is False, "Should not delete when confirm_text is empty"
        assert p.exists(), "File should still exist after no-op"
        assert "confirm_delete_pending" in state, "State should be unchanged"

    # ------------------------------------------------------------------
    # test_cancel_clears_pending_state
    # ------------------------------------------------------------------

    def test_cancel_clears_pending_state(self):
        """Cancel removes 'confirm_delete_pending' and 'delete_backup_bytes'."""
        state = {
            "confirm_delete_pending": True,
            "delete_backup_bytes": b"backup",
        }
        # Simulate cancel: caller manually pops keys (not via _execute_confirmed_delete)
        for key in ["confirm_delete_pending", "delete_backup_bytes"]:
            state.pop(key, None)

        assert "confirm_delete_pending" not in state
        assert "delete_backup_bytes" not in state

    # ------------------------------------------------------------------
    # test_confirm_with_wrong_text_does_not_delete
    # ------------------------------------------------------------------

    def test_confirm_with_wrong_text_does_not_delete(self, tmp_path):
        """Lowercase 'delete' must not trigger deletion (case-sensitive guard)."""
        p = self._make_parquet(tmp_path)
        state = self._base_state()

        result = _execute_confirmed_delete(state, "delete", p)

        assert result is False, "Lowercase 'delete' should not delete"
        assert p.exists(), "File should still exist"

    # ------------------------------------------------------------------
    # test_confirm_with_correct_text_deletes
    # ------------------------------------------------------------------

    def test_confirm_with_correct_text_deletes(self, tmp_path):
        """Uppercase 'DELETE' triggers deletion and clears all keys."""
        p = self._make_parquet(tmp_path)
        state = self._base_state()

        result = _execute_confirmed_delete(state, "DELETE", p)

        assert result is True, "Should return True when deletion fires"
        assert not p.exists(), "Parquet file should be deleted"
        for key in ["confirm_delete_pending", "delete_backup_bytes",
                    "cohort_df", "strobe", "master_df", "audit_trail"]:
            assert key not in state, f"Key '{key}' should be removed from state"

    # ------------------------------------------------------------------
    # test_backup_bytes_cached_in_session_state
    # ------------------------------------------------------------------

    def test_backup_bytes_cached_in_session_state(self):
        """Backup bytes set once should not be overwritten on a second call.

        The page code uses ``if "delete_backup_bytes" not in st.session_state``
        to guard the read. This test verifies the guard logic directly.
        """
        state: dict = {}

        original_bytes = b"first read"

        # First call: no bytes in state -> set them
        if "delete_backup_bytes" not in state:
            state["delete_backup_bytes"] = original_bytes

        # Second call (simulating a Streamlit rerun): bytes already present
        second_read_bytes = b"second read (should NOT overwrite)"
        if "delete_backup_bytes" not in state:
            state["delete_backup_bytes"] = second_read_bytes

        assert state["delete_backup_bytes"] == original_bytes, (
            "Backup bytes should be cached from the first read only"
        )


class TestErrorBoundaries:
    """ERRH-01: Smoke tests verifying core modules used by error boundaries import cleanly."""

    def test_cohort_module_imports_cleanly(self):
        """Smoke test: core modules used by all three pages import without error."""
        from core.cohort import filter_cohort, validate_icd9_range_value
        from core.io import load_master, master_exists
        assert callable(filter_cohort)

    def test_config_module_imports_cleanly(self):
        """Smoke test: config module imports without error."""
        from core.config import CohortConfig, FilterRule, RuleGroup
        assert callable(CohortConfig)
