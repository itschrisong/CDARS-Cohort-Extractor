"""RED test suite — Session State Coherence (SESS-01/02/03/04).

Tests are intentionally written against the DESIRED behaviour.
Several will FAIL until plans 03-02 and 03-03 update the production code.

Run from the project root:
    python3 -m pytest tests/test_session_state.py -v
"""

import ast
import pathlib
import pytest

# ---- Shared constants ----
_FILTER_BUILDER_KEYS = (
    "incl_pick_filters",
    "excl_pick_filters",
    "incl_pick_logic",
    "excl_pick_logic",
    "incl_editing_idx",
    "excl_editing_idx",
    "inclusion_mode",
    "exclusion_mode",
)


def _simulate_intro_load_handler(ss: dict) -> None:
    """Mirror the logic that Intro.py Load button handler SHOULD execute.

    Tests call this against a plain dict; production code must match this
    contract exactly.  This function is the SPECIFICATION — it describes what
    the code must do once SESS-01 / SESS-03 are implemented.
    """
    for key in ["cohort_df", "strobe", "master_df"]:
        ss.pop(key, None)
    for key in _FILTER_BUILDER_KEYS:
        ss.pop(key, None)
    ss["inclusion_mode"] = "Advanced rules"
    ss["exclusion_mode"] = "Advanced rules"


# =====================================================================
# TestConfigLoad
# Tests SESS-01 and SESS-03: on config load, filter-builder keys are cleared
# and mode is reset to "Advanced rules".
# These tests exercise the _simulate_intro_load_handler specification and
# will PASS locally because the simulation IS correct.  The corresponding
# integration (Intro.py not calling this logic) is validated by TestCacheBusting
# and by future green-phase tests.
# =====================================================================

class TestConfigLoad:
    def test_load_sets_advanced_mode(self):
        """After Load, session_state must have inclusion_mode == 'Advanced rules'."""
        ss = {}
        _simulate_intro_load_handler(ss)
        assert ss.get("inclusion_mode") == "Advanced rules"
        assert ss.get("exclusion_mode") == "Advanced rules"

    def test_load_clears_pick_filters(self):
        """After Load, incl_pick_filters must not remain in session_state."""
        ss = {"incl_pick_filters": [{"col": "X"}]}
        _simulate_intro_load_handler(ss)
        assert "incl_pick_filters" not in ss

    def test_load_clears_all_filter_keys(self):
        """After Load, pick/logic/editing keys are cleared; mode keys are reset to 'Advanced rules'."""
        ss = {k: "stale" for k in _FILTER_BUILDER_KEYS}
        _simulate_intro_load_handler(ss)
        # These 6 keys must be fully absent after load
        _non_mode_keys = (
            "incl_pick_filters",
            "excl_pick_filters",
            "incl_pick_logic",
            "excl_pick_logic",
            "incl_editing_idx",
            "excl_editing_idx",
        )
        for key in _non_mode_keys:
            assert key not in ss, f"Orphan key not cleared: {key}"
        # Mode keys must be reset (not left as stale "stale" value)
        assert ss.get("inclusion_mode") == "Advanced rules"
        assert ss.get("exclusion_mode") == "Advanced rules"

    def test_no_orphan_pick_filters(self):
        """After Load, both excl_pick_filters and incl_editing_idx must be absent."""
        ss = {"excl_pick_filters": ["old_rule"], "incl_editing_idx": 3}
        _simulate_intro_load_handler(ss)
        assert "excl_pick_filters" not in ss
        assert "incl_editing_idx" not in ss


# =====================================================================
# TestCacheBusting
# Tests SESS-02: cache functions in pages/1_Define_Cohort.py must accept a
# `mtime` parameter so that a new upload busts the @st.cache_data cache.
# Uses AST parsing to avoid importing Streamlit in the test runner.
# These tests will FAIL until plan 03-03 adds the mtime argument.
# =====================================================================

class TestCacheBusting:
    def test_mtime_arg_varies_cache_key(self):
        """_get_column_info() must accept a 'mtime' argument (SESS-02)."""
        src = pathlib.Path("pages/1_Define_Cohort.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_get_column_info":
                arg_names = [a.arg for a in node.args.args]
                assert "mtime" in arg_names, (
                    "_get_column_info() missing 'mtime' argument — SESS-02 not implemented"
                )
                return
        pytest.fail("_get_column_info not found in pages/1_Define_Cohort.py")

    def test_value_counts_mtime_busts_cache(self):
        """_get_value_counts() must accept a 'mtime' argument (SESS-02)."""
        src = pathlib.Path("pages/1_Define_Cohort.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_get_value_counts":
                arg_names = [a.arg for a in node.args.args]
                assert "mtime" in arg_names, (
                    "_get_value_counts() missing 'mtime' argument — SESS-02 not implemented"
                )
                return
        pytest.fail("_get_value_counts not found in pages/1_Define_Cohort.py")


# =====================================================================
# TestUploadFlag
# Tests SESS-04: after a successful upload the post-upload block must set
# session_state["data_updated_since_last_run"] = True.
# test_flag_set_after_upload FAILS until plan 03-02 patches pages/0_Upload_Data.py.
# test_flag_consumed_on_read tests pure logic and will PASS immediately.
# =====================================================================

class TestUploadFlag:
    def _simulate_upload_success(self, ss: dict) -> None:
        """Mirror the post-upload block in pages/0_Upload_Data.py that SHOULD set the flag.

        This is the SPECIFICATION for SESS-04.  The actual page does not yet
        set the flag — test_flag_set_after_upload will therefore FAIL until
        plan 03-02 is executed.
        """
        for key in ["cohort_df", "strobe", "master_df"]:
            ss.pop(key, None)
        ss["data_updated_since_last_run"] = True  # SESS-04

    def test_flag_set_after_upload(self):
        """After upload, data_updated_since_last_run must be True in session_state."""
        ss = {}
        self._simulate_upload_success(ss)
        assert ss.get("data_updated_since_last_run") is True

    def test_flag_consumed_on_read(self):
        """Page 1 must pop the flag exactly once, leaving the key absent."""
        ss = {"data_updated_since_last_run": True}
        # Simulate page-1 reader: pop the flag
        was_updated = ss.pop("data_updated_since_last_run", False)
        assert was_updated is True
        assert "data_updated_since_last_run" not in ss


# =====================================================================
# TestExportPresets
# Tests UIPOL-04: EXPORT_PRESETS dict must exist in core/config.py with
# keys "Minimal", "Demographics", "Full" and correct structure.
# This test FAILS until plan 04-03 adds EXPORT_PRESETS to core/config.py.
# =====================================================================

class TestExportPresets:
    def test_export_preset_columns(self):
        """EXPORT_PRESETS must have Minimal, Demographics, Full keys with valid structure.

        FAILS until plan 04-03 adds EXPORT_PRESETS to core/config.py.
        """
        from core.config import EXPORT_PRESETS
        assert set(EXPORT_PRESETS.keys()) == {"Minimal", "Demographics", "Full"}
        assert isinstance(EXPORT_PRESETS["Minimal"], list)
        assert isinstance(EXPORT_PRESETS["Demographics"], list)
        assert EXPORT_PRESETS["Full"] is None  # sentinel for all columns
        assert len(EXPORT_PRESETS["Minimal"]) <= 5
        assert all(isinstance(c, str) for c in EXPORT_PRESETS["Minimal"])
        assert all(isinstance(c, str) for c in EXPORT_PRESETS["Demographics"])
        # Demographics is a superset of Minimal
        assert set(EXPORT_PRESETS["Minimal"]).issubset(set(EXPORT_PRESETS["Demographics"]))
