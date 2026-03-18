"""Generic rule-based cohort filtering engine.

Replaces the old hardcoded self-harm pathways with a flexible operator-based
system that works for any cohort definition.

All string operators normalize whitespace (collapse runs of spaces/NBSP to
single space) before comparison — this handles the known CDARS data quality
issue where values like "2 Self-harm" and "2  Self-harm" coexist.
"""

import re
import pandas as pd
import numpy as np

from core.icd9 import has_icd9_range
from core.normalize import normalize_ws_series as _normalize_ws, normalize_ws_scalar as _normalize_ws_scalar


# ---------------------------------------------------------------------------
# ICD-9 range input validation
# ---------------------------------------------------------------------------

_E_V_CODE_RE = re.compile(r'^[EeVv]\d', re.IGNORECASE)


def validate_icd9_range_value(value: str) -> "str | None":
    """Validate an icd9_range operator value string.

    Returns None if valid, or an error message string if invalid.
    Empty string returns None (user may not have typed yet).

    Valid format: one or more comma-separated pairs of numeric bounds,
    each pair joined by a hyphen (e.g. '290-319.99' or '290-319.99, 410-410.99').
    E-codes (E800-E999) and V-codes (V01-V99) cannot be used with icd9_range —
    use the starts_with operator instead.
    """
    if not value or not value.strip():
        return None
    for pair in value.split(","):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split("-")
        if len(parts) != 2:
            return (
                f"Invalid range '{pair}': expected format like '290-319.99'. "
                "Each range must be two numbers separated by a hyphen."
            )
        lo, hi = parts[0].strip(), parts[1].strip()
        if _E_V_CODE_RE.match(lo) or _E_V_CODE_RE.match(hi):
            return (
                f"E-codes and V-codes (e.g. '{lo}') cannot be used with icd9_range. "
                "Use the starts_with operator instead "
                "(e.g. value 'E95' matches E950\u2013E959)."
            )
        try:
            float(lo)
            float(hi)
        except ValueError:
            return f"Range bounds must be numbers, got '{lo}' and '{hi}'."
    return None


# ---------------------------------------------------------------------------
# Operator functions: (Series, value, case_sensitive) -> boolean Series
# ---------------------------------------------------------------------------

def _op_starts_with(s, v, cs):
    series = _normalize_ws(s.astype(str))
    v = _normalize_ws_scalar(v)
    if not cs:
        series = series.str.lower()
        v = v.lower()
    return series.str.startswith(v, na=False)


def _op_contains(s, v, cs):
    series = _normalize_ws(s.astype(str))
    v = _normalize_ws_scalar(v)
    if not cs:
        series = series.str.lower()
        v = v.lower()
    return series.str.contains(v, na=False, regex=False)


def _op_equals(s, v, cs):
    series = _normalize_ws(s.astype(str))
    v = _normalize_ws_scalar(v)
    if not cs:
        series = series.str.lower()
        v = v.lower()
    return series == v


def _op_not_equals(s, v, cs):
    return ~_op_equals(s, v, cs)


def _op_in_list(s, v, cs):
    items = [_normalize_ws_scalar(x) for x in v.split(",")]
    series = _normalize_ws(s.astype(str))
    if not cs:
        series = series.str.lower()
        items = [x.lower() for x in items]
    return series.isin(items)


def _op_regex(s, v, cs):
    series = _normalize_ws(s.astype(str))
    if not cs:
        series = series.str.lower()
        v = v.lower()
    return series.str.contains(v, na=False, regex=True)


def _op_before_date(s, v, _):
    return pd.to_datetime(s, errors="coerce") < pd.Timestamp(v)


def _op_after_date(s, v, _):
    return pd.to_datetime(s, errors="coerce") > pd.Timestamp(v)


def _op_gt(s, v, _):
    return pd.to_numeric(s, errors="coerce") > float(v)


def _op_lt(s, v, _):
    return pd.to_numeric(s, errors="coerce") < float(v)


def _op_gte(s, v, _):
    return pd.to_numeric(s, errors="coerce") >= float(v)


def _op_lte(s, v, _):
    return pd.to_numeric(s, errors="coerce") <= float(v)


def _op_is_null(s, *_):
    str_s = _normalize_ws(s.astype(str)).str.lower()
    return s.isna() | str_s.isin(["", "nan", "none"])


def _op_not_null(s, *_):
    return ~_op_is_null(s)


OPERATORS = {
    "starts_with": _op_starts_with,
    "contains": _op_contains,
    "equals": _op_equals,
    "not_equals": _op_not_equals,
    "in_list": _op_in_list,
    "regex": _op_regex,
    "before_date": _op_before_date,
    "after_date": _op_after_date,
    ">": _op_gt,
    "<": _op_lt,
    ">=": _op_gte,
    "<=": _op_lte,
    "is_null": _op_is_null,
    "not_null": _op_not_null,
    "icd9_range": None,  # handled specially in apply_rule
}


def apply_rule(df, rule) -> pd.Series:
    """Apply a single FilterRule to df, returning a boolean mask.

    When ``rule.columns`` (plural) is set, the rule fans out across all
    listed columns with OR logic — matching if *any* column satisfies the
    operator condition.
    """
    cols = rule.get_columns()
    cols = [c for c in cols if c in df.columns]

    if not cols:
        return pd.Series(False, index=df.index)

    if rule.operator == "icd9_range":
        # value is comma-separated pairs: "290-319.99, 296.2-296.39"
        err = validate_icd9_range_value(rule.value)
        if err:
            raise ValueError(err)
        ranges = []
        for pair in rule.value.split(","):
            parts = pair.strip().split("-")
            if len(parts) == 2:
                try:
                    ranges.append((float(parts[0].strip()), float(parts[1].strip())))
                except ValueError:
                    continue
        if not ranges:
            return pd.Series(False, index=df.index)
        return has_icd9_range(df, cols, ranges).astype(bool)

    op_fn = OPERATORS.get(rule.operator)
    if op_fn is None:
        return pd.Series(False, index=df.index)

    # OR across columns: match if ANY column satisfies the condition
    combined = pd.Series(False, index=df.index)
    for col in cols:
        combined = combined | op_fn(df[col], rule.value, rule.case_sensitive)
    return combined


def apply_rule_group(df, group) -> pd.Series:
    """Apply a RuleGroup (list of rules with AND/OR logic), returning a boolean mask."""
    if not group.rules:
        return pd.Series(False, index=df.index)

    masks = [apply_rule(df, rule) for rule in group.rules]

    if group.logic.upper() == "AND":
        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m
        return combined
    else:  # OR
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
        return combined


def filter_cohort(df, config):
    """Apply inclusion/exclusion rules and return filtered cohort with STROBE flow.

    Parameters
    ----------
    df : DataFrame – master all_admissions
    config : CohortConfig

    Returns
    -------
    (clean_cohort_df, strobe_list)
        strobe_list is a list of (step_name, count_removed, count_remaining) tuples
        for STROBE flow display.
    """
    # Unconditional copy — prevents mutation of caller's DataFrame.
    # Required for pandas 3.0 Copy-on-Write compatibility (CoW makes
    # conditional copies fragile; unconditional is always correct).
    df = df.copy()

    date_col = config.date_column
    if "attendance_date" not in df.columns:
        df["attendance_date"] = pd.to_datetime(df[date_col], errors="coerce")
    if "year" not in df.columns:
        df["year"] = df["attendance_date"].dt.year

    strobe = []
    total_rows = len(df)
    strobe.append(("Total records in master", 0, total_rows))

    # 1. Apply inclusion rules → cohort subset
    if config.inclusion.rules:
        inclusion_mask = apply_rule_group(df, config.inclusion)
        cohort = df.loc[inclusion_mask].copy()
    else:
        cohort = df.copy()

    strobe.append(("After inclusion filter", total_rows - len(cohort), len(cohort)))

    # 2. Apply exclusion rules sequentially (STROBE counting)
    if config.exclusion.rules:
        for i, rule in enumerate(config.exclusion.rules):
            before = len(cohort)
            excl_mask = apply_rule(cohort, rule)
            cohort = cohort.loc[~excl_mask].copy()
            removed = before - len(cohort)
            # Build a readable label — use rule.label if set, else auto-generate
            if hasattr(rule, 'label') and rule.label:
                label = rule.label
            else:
                cols = rule.get_columns()
                col_label = cols[0] if len(cols) == 1 else f"{len(cols)} columns"
                val_display = rule.value[:40] + "..." if len(rule.value) > 40 else rule.value
                label = f"Excl: {col_label} {rule.operator} {val_display}"
            strobe.append((label, removed, len(cohort)))

    # 3. Death buffer
    if config.death_buffer_days is not None:
        before = len(cohort)
        excl_death = pd.Series(False, index=cohort.index)

        if "Episode Death (Y/N)" in cohort.columns:
            excl_death = cohort["Episode Death (Y/N)"].astype(str).str.upper().str.strip() == "Y"

        if "Date of Registered Death" in cohort.columns:
            death_date = pd.to_datetime(cohort["Date of Registered Death"], errors="coerce")
            within_buffer = (death_date - cohort["attendance_date"]).dt.days <= config.death_buffer_days
            excl_death = excl_death | (within_buffer & death_date.notna())

        cohort = cohort.loc[~excl_death].copy()
        strobe.append((f"Excl: death within {config.death_buffer_days}d", before - len(cohort), len(cohort)))

    # 4. Follow-up buffer
    if config.followup_buffer_days is not None:
        before = len(cohort)
        last_date = cohort["attendance_date"].max()
        if pd.notna(last_date):
            cutoff = last_date - pd.Timedelta(days=config.followup_buffer_days)
            excl_followup = cohort["attendance_date"] > cutoff
            cohort = cohort.loc[~excl_followup].copy()
        strobe.append((f"Excl: incomplete follow-up ({config.followup_buffer_days}d)", before - len(cohort), len(cohort)))

    # 5. First presentation per patient
    pid_col = config.patient_id_column
    if config.first_presentation_only:
        before = len(cohort)
        cohort = cohort.sort_values([pid_col, "attendance_date"]).drop_duplicates(subset=[pid_col], keep="first")
        strobe.append(("Excl: keep first presentation per patient", before - len(cohort), len(cohort)))

    # Sort and assign IDs
    cohort = cohort.sort_values([pid_col, "attendance_date"]).reset_index(drop=True)
    cohort["presentation_id"] = np.arange(len(cohort))

    strobe.append(("Analysis cohort", 0, len(cohort)))

    return cohort, strobe
