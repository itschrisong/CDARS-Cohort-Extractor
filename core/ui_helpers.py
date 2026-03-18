"""Pure helper functions for UI state-machine logic.

Kept Streamlit-free so they can be unit-tested without importing the st module.
"""

from __future__ import annotations

import pathlib
from typing import Any, List, MutableMapping

import pandas as pd


# Session state keys cleared when cohort results are invalidated
COHORT_STATE_KEYS = ("cohort_df", "strobe", "master_df")


def _execute_confirmed_delete(
    session_state: MutableMapping[str, Any],
    confirm_text: str,
    parquet_path: "pathlib.Path",
) -> bool:
    """Execute a confirmed delete of the master parquet file.

    This is the pure state-machine core of the two-step delete flow.
    Returns True if deletion was performed, False otherwise.

    The caller is responsible for calling st.rerun() after a True return.

    Rules:
    - Returns False (no-op) unless confirm_text == "DELETE" exactly.
    - On success: deletes parquet_path if it exists, removes all cohort-related
      and confirmation-related keys from session_state.
    """
    if confirm_text != "DELETE":
        return False

    if parquet_path.exists():
        parquet_path.unlink()

    for key in [
        "cohort_df",
        "strobe",
        "master_df",
        "audit_trail",
        "confirm_delete_pending",
        "delete_backup_bytes",
    ]:
        session_state.pop(key, None)

    return True


def clear_cohort_state(session_state: MutableMapping[str, Any]) -> None:
    """Remove cached cohort results from session state."""
    for key in COHORT_STATE_KEYS:
        session_state.pop(key, None)


def year_bar_chart_data(cdf: pd.DataFrame) -> "pd.DataFrame | None":
    """Build presentations-per-year DataFrame from a cohort, or None if unavailable."""
    if "year" not in cdf.columns:
        return None
    try:
        return (
            cdf["year"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .rename_axis("Year")
            .reset_index(name="Presentations")
        )
    except (ValueError, TypeError):
        return None


def _detect_zero_match_columns(strobe: list, config) -> List[str]:
    """Return up to 3 column names from inclusion rules when inclusion filter matched 0 rows.

    Args:
        strobe: List of (step_name, removed, remaining) tuples from filter_cohort().
        config: CohortConfig whose inclusion.rules are inspected.

    Returns:
        List of up to 3 column names when "After inclusion filter" step has remaining == 0.
        Returns empty list when rows remain or the step is absent.
    """
    for step_name, _removed, remaining in strobe:
        if step_name == "After inclusion filter" and remaining == 0:
            cols: List[str] = []
            for rule in config.inclusion.rules:
                for col in rule.get_columns():
                    if col not in cols:
                        cols.append(col)
            return cols[:3]
    return []
