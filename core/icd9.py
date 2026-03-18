"""ICD-9-CM utility functions.

Ported from notebooks 01 and 03 of the self-harm model decay project.
"""

import pandas as pd
import numpy as np


def flag_codes(df, columns, prefix):
    """Return boolean Series: True if any column value starts with *prefix*.

    Uses str.startswith for correct ICD prefix matching (e.g. 'E95' matches
    'E950', 'E951.2' but not 'E9').
    """
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        mask = mask | df[col].astype(str).str.startswith(prefix, na=False)
    return mask


def has_icd9_range(df, columns, ranges):
    """Check if any diagnosis column contains ICD-9-CM codes in numeric ranges.

    Parameters
    ----------
    df : DataFrame
    columns : list of str – diagnosis code column names
    ranges : list of (float, float) – inclusive (start, end) numeric ranges

    Returns
    -------
    Series of int (0/1)
    """
    match = pd.Series(False, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        # Convert to numeric once per column, then check all ranges
        numeric = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
        for start, end in ranges:
            match = match | numeric.between(start, end)
    return match.astype(int)


# Charlson conditions with ICD-9 ranges and weights (Quan et al. 2005)
CHARLSON_CONDITIONS = {
    "MI": {"ranges": [(410, 410.99), (412, 412.99)], "weight": 1},
    "CHF": {"ranges": [(428, 428.99)], "weight": 1},
    "PVD": {"ranges": [(443.9, 443.9), (441, 441.99), (785.4, 785.4)], "weight": 1},
    "CVD": {"ranges": [(430, 438.99)], "weight": 1},
    "Dementia": {"ranges": [(290, 290.99)], "weight": 1},
    "COPD": {
        "ranges": [(490, 496.99), (500, 505.99), (506.4, 506.4)],
        "weight": 1,
    },
    "Rheumatic": {
        "ranges": [
            (710.0, 710.1),
            (710.4, 710.4),
            (714.0, 714.2),
            (714.81, 714.81),
            (725, 725.99),
        ],
        "weight": 1,
    },
    "PUD": {"ranges": [(531, 534.99)], "weight": 1},
    "Liver_mild": {
        "ranges": [(571.2, 571.2), (571.4, 571.49), (571.5, 571.5), (571.6, 571.6)],
        "weight": 1,
    },
    "DM_uncomplicated": {"ranges": [(250.0, 250.33), (250.7, 250.7)], "weight": 1},
    "DM_complicated": {"ranges": [(250.4, 250.69)], "weight": 2},
    "Hemiplegia": {"ranges": [(342, 342.99), (344.1, 344.1)], "weight": 2},
    "Renal": {
        "ranges": [(582, 582.99), (583, 583.77), (585, 586.99), (588, 588.99)],
        "weight": 2,
    },
    "Cancer": {
        "ranges": [(140, 172.99), (174, 195.89), (200, 208.99)],
        "weight": 2,
    },
    "Liver_severe": {"ranges": [(572.2, 572.8), (456.0, 456.21)], "weight": 3},
    "Metastatic": {"ranges": [(196, 199.99)], "weight": 6},
    "AIDS": {"ranges": [(42, 44.99)], "weight": 6},
}

# Pre-compute flat list of (condition, start, end, weight) for faster iteration
_CHARLSON_FLAT = []
for _cond, _info in CHARLSON_CONDITIONS.items():
    for _start, _end in _info["ranges"]:
        _CHARLSON_FLAT.append((_cond, _start, _end, _info["weight"]))


def compute_charlson(code_list):
    """Compute Charlson Comorbidity Index from a list of ICD-9-CM code strings.

    Applies Quan et al. 2005 weights with hierarchy rules:
    - DM complicated > uncomplicated
    - Liver severe > mild
    - Metastatic > Cancer
    """
    score = 0
    conditions_found = set()

    # Pre-convert all codes to float once
    numerics = []
    for code_str in code_list:
        try:
            numerics.append(float(code_str))
        except (ValueError, TypeError):
            continue

    for numeric in numerics:
        for cond, start, end, weight in _CHARLSON_FLAT:
            if cond in conditions_found:
                continue
            if start <= numeric <= end:
                conditions_found.add(cond)
                score += weight
                # Found this condition, no need to check more ranges for it
                break

    # Hierarchy rules: don't double-count
    if "DM_complicated" in conditions_found and "DM_uncomplicated" in conditions_found:
        score -= CHARLSON_CONDITIONS["DM_uncomplicated"]["weight"]
    if "Liver_severe" in conditions_found and "Liver_mild" in conditions_found:
        score -= CHARLSON_CONDITIONS["Liver_mild"]["weight"]
    if "Metastatic" in conditions_found and "Cancer" in conditions_found:
        score -= CHARLSON_CONDITIONS["Cancer"]["weight"]

    return score


def compute_charlson_vectorized(codes_df, patient_col, date_col, diag_cols, index_dates_df):
    """Vectorized Charlson computation for all patients at once.

    Parameters
    ----------
    codes_df : DataFrame with patient_col, date_col, and diag_cols
    patient_col : str
    date_col : str
    diag_cols : list of str
    index_dates_df : DataFrame with patient_col, date_col (index presentation dates),
                     and '_cohort_row' column mapping back to cohort row index

    Returns
    -------
    Series indexed by _cohort_row with Charlson scores
    """
    # Melt to long format
    dx_long = codes_df[[patient_col, date_col] + diag_cols].melt(
        id_vars=[patient_col, date_col],
        value_vars=diag_cols,
        value_name="code",
    ).dropna(subset=["code"])

    # Convert codes to numeric
    dx_long["numeric"] = pd.to_numeric(dx_long["code"].astype(str).str.strip(), errors="coerce")
    dx_long = dx_long.dropna(subset=["numeric"])

    if dx_long.empty:
        return pd.Series(0, index=index_dates_df["_cohort_row"], dtype=np.int16)

    # Flag each Charlson condition vectorially
    for cond, info in CHARLSON_CONDITIONS.items():
        cond_mask = pd.Series(False, index=dx_long.index)
        for start, end in info["ranges"]:
            cond_mask = cond_mask | dx_long["numeric"].between(start, end)
        dx_long[f"_c_{cond}"] = cond_mask

    cond_cols = [f"_c_{c}" for c in CHARLSON_CONDITIONS]

    # Join with index dates to filter prior-only diagnoses
    # Use merge to get all (patient, dx_date, index_date) combos
    dx_long = dx_long.merge(
        index_dates_df[[patient_col, date_col, "_cohort_row"]].rename(
            columns={date_col: "_idx_date"}
        ),
        on=patient_col,
        how="inner",
    )

    # Keep only prior diagnoses
    dx_long = dx_long[dx_long[date_col] < dx_long["_idx_date"]]

    if dx_long.empty:
        return pd.Series(0, index=index_dates_df["_cohort_row"], dtype=np.int16)

    # For each (cohort_row, condition), check if ANY prior diagnosis matches
    # Group by _cohort_row and take max of each condition flag
    per_presentation = dx_long.groupby("_cohort_row")[cond_cols].any()

    # Compute weighted score
    weights = pd.Series(
        {f"_c_{c}": info["weight"] for c, info in CHARLSON_CONDITIONS.items()}
    )
    scores = (per_presentation * weights).sum(axis=1).astype(np.int16)

    # Apply hierarchy rules
    if "_c_DM_complicated" in per_presentation.columns and "_c_DM_uncomplicated" in per_presentation.columns:
        both_dm = per_presentation["_c_DM_complicated"] & per_presentation["_c_DM_uncomplicated"]
        scores[both_dm] -= CHARLSON_CONDITIONS["DM_uncomplicated"]["weight"]

    if "_c_Liver_severe" in per_presentation.columns and "_c_Liver_mild" in per_presentation.columns:
        both_liver = per_presentation["_c_Liver_severe"] & per_presentation["_c_Liver_mild"]
        scores[both_liver] -= CHARLSON_CONDITIONS["Liver_mild"]["weight"]

    if "_c_Metastatic" in per_presentation.columns and "_c_Cancer" in per_presentation.columns:
        both_cancer = per_presentation["_c_Metastatic"] & per_presentation["_c_Cancer"]
        scores[both_cancer] -= CHARLSON_CONDITIONS["Cancer"]["weight"]

    # Reindex to all cohort rows (fill missing with 0)
    return scores.reindex(index_dates_df["_cohort_row"], fill_value=0)
