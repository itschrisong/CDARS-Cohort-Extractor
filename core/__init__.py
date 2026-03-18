"""DoEMDataWrangler core API."""

from core.config import CohortConfig, FilterRule, RuleGroup, COLUMN_PRESETS
from core.cohort import apply_rule, apply_rule_group, filter_cohort, OPERATORS
from core.features import build_features, assign_eras
from core.io import load_master, master_exists, master_info


def run_pipeline(config, master_df=None, progress_cb=None):
    """Run full cohort extraction pipeline with given config.

    Parameters
    ----------
    config : CohortConfig
    master_df : DataFrame, optional – pre-loaded master parquet
    progress_cb : callable(stage, pct), optional

    Returns
    -------
    dict with keys: cohort_df, strobe_flow
    """
    if master_df is None:
        master_df = load_master()

    if progress_cb:
        progress_cb("filtering", 0.0)

    cohort_df, strobe = filter_cohort(master_df, config)

    if progress_cb:
        progress_cb("features", 0.3)

    any_features = any(config.features.values())
    if any_features or config.eras:
        cohort_df = build_features(cohort_df, master_df, config, progress_cb=progress_cb)

    if progress_cb:
        progress_cb("done", 1.0)

    return {"cohort_df": cohort_df, "strobe_flow": strobe}
