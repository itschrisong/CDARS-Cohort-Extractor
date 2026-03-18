"""Feature engineering — opt-in dispatcher with index-level, lookback, and recurrence.

All features are opt-in via config.features dict. Recurrence (formerly outcome.py)
is now an optional feature rather than a separate pipeline step.
"""

import numpy as np
import pandas as pd

from core.icd9 import has_icd9_range, compute_charlson_vectorized
from core.cohort import apply_rule, apply_rule_group
from core.config import FilterRule, COLUMN_PRESETS


def _get_all_diag_cols(df):
    """Return the list of all diagnosis code columns present in df."""
    candidates = COLUMN_PRESETS["All diagnosis codes (A&E + IP)"]
    return [c for c in candidates if c in df.columns]


def build_features(cohort_df, master_df, config, progress_cb=None):
    """Single entry point for all opt-in features.

    Parameters
    ----------
    cohort_df : DataFrame – analysis cohort (post-filtering)
    master_df : DataFrame – full master parquet
    config : CohortConfig
    progress_cb : callable(stage, pct), optional

    Returns
    -------
    DataFrame – cohort with requested feature columns added.
    """
    df = cohort_df.copy()
    features = config.features
    pid_col = config.patient_id_column
    n = len(df)

    # Ensure attendance_date on both frames
    if "attendance_date" not in df.columns:
        df["attendance_date"] = pd.to_datetime(
            df[config.date_column], errors="coerce"
        )
    if "attendance_date" not in master_df.columns:
        master_df = master_df.copy()
        master_df["attendance_date"] = pd.to_datetime(
            master_df[config.date_column], errors="coerce"
        )

    if progress_cb:
        progress_cb("features", 0.05)

    # --- Demographics ---
    if features.get("demographics"):
        df["feat_age"] = pd.to_numeric(
            df["Admission Age (Year) (episode based)"], errors="coerce"
        ).astype(float)
        df["feat_female"] = (
            df["Sex"].astype(str).str.strip().str.upper() == "F"
        ).astype(int)

    if progress_cb:
        progress_cb("features", 0.15)

    # --- Triage ---
    if features.get("triage"):
        df["feat_triage"] = pd.to_numeric(df["Triage Category"], errors="coerce")
        triage_median = df["feat_triage"].median()
        if pd.isna(triage_median):
            triage_median = 3
        df["feat_triage"] = df["feat_triage"].fillna(triage_median)

    if progress_cb:
        progress_cb("features", 0.20)

    # --- ICD-9 flag definitions ---
    if features.get("icd9_flags") and config.icd9_flag_definitions:
        diag_cols = _get_all_diag_cols(df)
        for feat_name, ranges in config.icd9_flag_definitions.items():
            df[f"feat_{feat_name}"] = has_icd9_range(df, diag_cols, ranges)

    if progress_cb:
        progress_cb("features", 0.25)

    # --- Custom index features (binary flags from any column) ---
    for feat_def in config.custom_features:
        feat_name = feat_def.get("name", "")
        feat_col = feat_def.get("column", "")
        feat_op = feat_def.get("operator", "equals")
        feat_val = feat_def.get("value", "")
        if not feat_name or not feat_col:
            continue
        if feat_col not in df.columns:
            df[f"feat_{feat_name}"] = np.int8(0)
            continue
        rule = FilterRule(column=feat_col, operator=feat_op, value=feat_val)
        df[f"feat_{feat_name}"] = apply_rule(df, rule).astype(np.int8)

    if progress_cb:
        progress_cb("features", 0.30)

    # --- Lookback counts ---
    if features.get("lookback_counts"):
        df = _build_lookback_features(df, master_df, config, progress_cb)

    if progress_cb:
        progress_cb("features", 0.70)

    # --- Charlson ---
    if features.get("charlson"):
        df = _build_charlson(df, master_df, config)

    if progress_cb:
        progress_cb("features", 0.80)

    # --- Recurrence ---
    if features.get("recurrence"):
        df = _compute_recurrence(df, master_df, config, progress_cb)

    if progress_cb:
        progress_cb("features", 0.95)

    # --- Eras ---
    if config.eras:
        df = assign_eras(df, config.eras)

    if progress_cb:
        progress_cb("features_done", 1.0)

    return df


# ---------------------------------------------------------------------------
# Internal feature builders
# ---------------------------------------------------------------------------

def _vectorized_lookback_counts(cohort_ref, cohort_ns, event_df, key_col, date_col, lookback_ns):
    """Vectorized lookback using numpy searchsorted per patient group.

    Returns (lifetime_count, lookback_count, days_since_last) arrays.
    """
    n = len(cohort_ref)
    lifetime = np.zeros(n, dtype=np.int32)
    recent = np.zeros(n, dtype=np.int32)
    days_since = np.full(n, np.nan)
    ns_per_day = np.float64(86_400_000_000_000)

    event_sorted = event_df[[key_col, date_col]].dropna(subset=[date_col]).copy()
    event_sorted = event_sorted.sort_values([key_col, date_col])
    event_sorted["_ns"] = event_sorted[date_col].values.astype("int64")

    patient_arrays = {}
    for pid, grp in event_sorted.groupby(key_col):
        patient_arrays[pid] = grp["_ns"].values

    cohort_frame = pd.DataFrame({
        "_ref": cohort_ref,
        "_ns": cohort_ns,
        "_row": np.arange(n),
    })

    for pid, grp in cohort_frame.groupby("_ref"):
        arr = patient_arrays.get(pid)
        if arr is None:
            continue

        rows = grp["_row"].values
        idx_ns = grp["_ns"].values

        pos = np.searchsorted(arr, idx_ns, side="left")
        lifetime[rows] = pos

        has_prior = pos > 0
        if has_prior.any():
            prior_rows = rows[has_prior]
            prior_pos = pos[has_prior]
            last_ns = arr[prior_pos - 1]
            days_since[prior_rows] = (idx_ns[has_prior] - last_ns) / ns_per_day

            cutoff_ns = idx_ns[has_prior] - lookback_ns
            cutoff_pos = np.searchsorted(arr, cutoff_ns, side="left")
            recent[prior_rows] = prior_pos - cutoff_pos

    return lifetime, recent, days_since


def _vectorized_window_count(cohort_ref, cohort_ns, event_df, key_col, date_col, lookback_ns):
    """Count events in [index - lookback, index) window per cohort row."""
    n = len(cohort_ref)
    counts = np.zeros(n, dtype=np.int32)

    event_sorted = event_df[[key_col, date_col]].dropna(subset=[date_col]).copy()
    event_sorted = event_sorted.sort_values([key_col, date_col])
    event_sorted["_ns"] = event_sorted[date_col].values.astype("int64")

    patient_arrays = {}
    for pid, grp in event_sorted.groupby(key_col):
        patient_arrays[pid] = grp["_ns"].values

    cohort_frame = pd.DataFrame({
        "_ref": cohort_ref,
        "_ns": cohort_ns,
        "_row": np.arange(n),
    })

    for pid, grp in cohort_frame.groupby("_ref"):
        arr = patient_arrays.get(pid)
        if arr is None:
            continue

        rows = grp["_row"].values
        idx_ns = grp["_ns"].values
        cutoff_ns = idx_ns - lookback_ns

        hi = np.searchsorted(arr, idx_ns, side="left")
        lo = np.searchsorted(arr, cutoff_ns, side="left")
        counts[rows] = hi - lo

    return counts


def _vectorized_any_in_window(cohort_ref, cohort_ns, event_df, key_col, date_col, lookback_ns):
    """Binary flag: any event in [index - lookback, index) window."""
    counts = _vectorized_window_count(cohort_ref, cohort_ns, event_df, key_col, date_col, lookback_ns)
    return (counts > 0).astype(np.int8)


def _build_lookback_features(df, master_df, config, progress_cb=None):
    """Compute lookback features from patient history.

    Instead of hardcoded sh_cohort, re-applies the inclusion rules to master
    to identify which historical events count as prior cohort events.
    """
    pid_col = config.patient_id_column
    n = len(df)
    lookback = config.lookback_days
    ns_per_day = np.int64(86_400_000_000_000)
    lookback_ns = np.int64(lookback) * ns_per_day

    ref_keys = df[pid_col].values
    idx_ns = df["attendance_date"].values.astype("int64")

    if progress_cb:
        progress_cb("lookback_events", 0.35)

    # --- Prior cohort events (re-apply inclusion rules to master) ---
    if config.inclusion.rules:
        cohort_mask = apply_rule_group(master_df, config.inclusion)
        cohort_sub = master_df.loc[cohort_mask, [pid_col, "attendance_date"]]
    else:
        cohort_sub = master_df[[pid_col, "attendance_date"]]

    lifetime, recent, days_since = _vectorized_lookback_counts(
        ref_keys, idx_ns, cohort_sub, pid_col, "attendance_date", lookback_ns
    )
    df["feat_prior_events_lifetime"] = lifetime
    df[f"feat_prior_events_{lookback}d"] = recent
    df["feat_days_since_last_event"] = days_since

    if progress_cb:
        progress_cb("lookback_ed", 0.50)

    # --- Filter master to cohort patients only ---
    cohort_patients = set(df[pid_col].unique())
    patient_master = master_df[master_df[pid_col].isin(cohort_patients)]

    # --- ED visits in lookback ---
    df[f"feat_ed_visits_{lookback}d"] = _vectorized_window_count(
        ref_keys, idx_ns, patient_master, pid_col, "attendance_date", lookback_ns
    )

    if progress_cb:
        progress_cb("lookback_custom", 0.60)

    # --- Custom lookback flags ---
    for flag_def in config.custom_lookback_flags:
        flag_name = flag_def.get("name", "")
        flag_col = flag_def.get("column", "")
        flag_op = flag_def.get("operator", "equals")
        flag_val = flag_def.get("value", "")
        flag_window = flag_def.get("window_days")
        if not flag_name or not flag_col:
            continue

        window_ns = np.int64(flag_window or lookback) * ns_per_day
        rule = FilterRule(column=flag_col, operator=flag_op, value=flag_val)
        if flag_col in patient_master.columns:
            match_mask = apply_rule(patient_master, rule)
            match_sub = patient_master.loc[match_mask, [pid_col, "attendance_date"]]
            df[f"feat_{flag_name}"] = _vectorized_any_in_window(
                ref_keys, idx_ns, match_sub, pid_col, "attendance_date", window_ns
            )
        else:
            df[f"feat_{flag_name}"] = np.int8(0)

    return df


def _build_charlson(df, master_df, config):
    """Compute Charlson comorbidity index."""
    pid_col = config.patient_id_column
    n = len(df)
    cohort_patients = set(df[pid_col].unique())
    patient_master = master_df[master_df[pid_col].isin(cohort_patients)]
    diag_cols = _get_all_diag_cols(patient_master)

    if diag_cols:
        index_dates_df = pd.DataFrame({
            pid_col: df[pid_col].values,
            "attendance_date": df["attendance_date"].values,
            "_cohort_row": np.arange(n),
        })
        pm_for_charlson = patient_master[[pid_col, "attendance_date"] + diag_cols]
        df["feat_charlson"] = compute_charlson_vectorized(
            pm_for_charlson, pid_col, "attendance_date",
            diag_cols, index_dates_df
        ).values
    else:
        df["feat_charlson"] = np.int16(0)

    return df


def _compute_recurrence(df, master_df, config, progress_cb=None):
    """Compute recurrence outcomes (was outcome.py).

    For each cohort row, find the next event by the same patient and compute
    whether it falls within each recurrence window.
    """
    pid_col = config.patient_id_column
    windows = config.recurrence_windows
    n = len(df)

    if progress_cb:
        progress_cb("recurrence", 0.82)

    # Get all cohort-matching events from master for outcome lookup
    if config.inclusion.rules:
        cohort_mask = apply_rule_group(master_df, config.inclusion)
        all_events = master_df.loc[
            cohort_mask, [pid_col, "attendance_date"]
        ].copy()
    else:
        all_events = master_df[[pid_col, "attendance_date"]].copy()

    all_events = all_events.dropna(subset=["attendance_date"])
    all_events = all_events.sort_values([pid_col, "attendance_date"]).reset_index(drop=True)

    days_to_next = np.full(n, np.nan)
    recurrence = {w: np.zeros(n, dtype=np.int8) for w in windows}
    ns_per_day = np.int64(86_400_000_000_000)

    # Build patient → sorted int64 ns arrays
    patient_dates = {}
    for pid, dates in all_events.groupby(pid_col)["attendance_date"]:
        patient_dates[pid] = dates.values.astype("int64")

    idx_ref = df[pid_col].values
    idx_ns = df["attendance_date"].values.astype("int64")

    for pid, group_idx in df.groupby(pid_col).groups.items():
        sh_dates = patient_dates.get(pid)
        if sh_dates is None:
            continue

        positions = group_idx.values if hasattr(group_idx, 'values') else np.array(list(group_idx))
        patient_idx_ns = idx_ns[positions]

        insert_pos = np.searchsorted(sh_dates, patient_idx_ns, side="right")

        valid = insert_pos < len(sh_dates)
        if not valid.any():
            continue

        valid_positions = positions[valid]
        valid_insert = insert_pos[valid]
        gaps_ns = sh_dates[valid_insert] - patient_idx_ns[valid]
        gaps_days = gaps_ns.astype(np.float64) / ns_per_day

        days_to_next[valid_positions] = gaps_days
        for w in windows:
            mask = gaps_days <= w
            recurrence[w][valid_positions[mask]] = 1

    if progress_cb:
        progress_cb("recurrence", 0.90)

    for w in windows:
        df[f"recurrence_{w}d"] = recurrence[w]
    df["days_to_next_event"] = days_to_next

    return df


def assign_eras(cohort_df, era_dict):
    """Assign training era labels based on year."""
    df = cohort_df.copy()
    if "year" not in df.columns:
        df["year"] = df["attendance_date"].dt.year
    year_to_era = {}
    for era_label, years in era_dict.items():
        for y in years:
            year_to_era[y] = era_label
    df["era"] = df["year"].map(year_to_era).fillna("unknown")
    return df
