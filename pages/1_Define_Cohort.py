"""Page 1 — Define Cohort: rule-based cohort extraction from CDARS A&E data."""

import streamlit as st
import pandas as pd

from core.config import (
    CohortConfig, FilterRule, RuleGroup,
    COLUMN_PRESETS, OPERATOR_HELP, OPERATORS_FOR_DTYPE,
)
from core.cohort import filter_cohort, OPERATORS, validate_icd9_range_value
from core.io import load_master, master_exists, MASTER_PARQUET
from core.ui_helpers import _detect_zero_match_columns, year_bar_chart_data

st.header("1. Define Cohort")

if not master_exists():
    st.error("No data loaded yet. Go to **0. Upload Data** and upload files first.")
    st.stop()

# Stale-cohort banner (SESS-04): shows once after new upload, consumed by .pop()
if st.session_state.pop("data_updated_since_last_run", False):
    st.warning(
        "New data has been uploaded since this cohort was last run. "
        "Click **Run Filter** to extract a fresh cohort from the updated dataset."
    )

# --- Get or init config ---
if "config" not in st.session_state:
    st.session_state["config"] = CohortConfig()
cfg = st.session_state["config"]


# =====================================================================
# Column metadata
# =====================================================================

@st.cache_data
def _get_column_info(mtime: float = 0.0):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(MASTER_PARQUET)
    schema = pf.schema_arrow
    info = {}
    for f in schema:
        pa_type = str(f.type)
        if "float" in pa_type or "int" in pa_type:
            dtype = "numeric"
        elif "timestamp" in pa_type or "date" in pa_type:
            dtype = "date"
        else:
            dtype = "text"
        if "Diagnosis" in f.name and "Description" not in f.name and "HAMDCT" not in f.name:
            dtype = "diagnosis"
        info[f.name] = dtype
    return info


@st.cache_data(show_spinner=False)
def _get_value_counts(column: str, top_n: int = 500, mtime: float = 0.0):
    s = pd.read_parquet(MASTER_PARQUET, columns=[column])[column]
    vc = s.value_counts(dropna=True).head(top_n)
    total_unique = s.nunique()
    return vc, total_unique


@st.cache_data(show_spinner=False)
def _get_numeric_stats(column: str, mtime: float = 0.0):
    s = pd.read_parquet(MASTER_PARQUET, columns=[column])[column]
    num = pd.to_numeric(s, errors="coerce")
    return {
        "min": num.min(),
        "max": num.max(),
        "mean": num.mean(),
        "median": num.median(),
        "missing": int(num.isna().sum()),
    }


_parquet_mtime = MASTER_PARQUET.stat().st_mtime if MASTER_PARQUET.exists() else 0.0
col_info = _get_column_info(mtime=_parquet_mtime)
all_columns = list(col_info.keys())

# --- Plain-English column descriptions ---
COLUMN_DESCRIPTIONS = {
    "Reference Key": "Unique patient identifier across all CDARS records",
    "Institution (IPAS)": "Hospital code (e.g. PMH, QEH, PWH)",
    "AE Number": "Unique A&E attendance number for this visit",
    "Sex": "Patient sex (M / F)",
    "Admission Age (Year) (episode based)": "Patient age in years at time of this visit",
    "Paycode (at discharge)": "Payment category — e.g. NE = non-eligible (non-local residents)",
    "District of Residence (system code)": "Numeric code for patient's residential district",
    "District of Residence Description": "Name of residential district (e.g. KOWLOON CITY, CHINA, OVERSEAS)",
    "Race Description": "Patient ethnicity/race",
    "Date of Registered Death": "Date of death from death registry (if applicable)",
    "Exact date of death": "Exact date of death (if applicable)",
    "Death Cause (Main Cause)": "Primary cause of death (ICD code)",
    "Death Cause (Supplementary Cause)": "Secondary cause of death (ICD code)",
    "Attendance Date (yyyy-mm-dd)": "Date the patient attended A&E",
    "Attendance Date (Hour)": "Hour of attendance (0-23)",
    "Admission from OAH (Y/N)": "Whether patient was admitted from Old Age Home",
    "Ambulance Case (Y/N)": "Whether patient arrived by ambulance",
    "Attendance Specialty (EIS)": "Specialty code for the A&E attendance",
    "Discharge Date (yyyy-mm-dd)": "Date discharged from A&E",
    "Discharge Hour (00-23)": "Hour of discharge (0-23)",
    "Traumatic Type": "Type of trauma (if injury-related visit)",
    "Domestic Type": "Type of domestic incident (if applicable)",
    "Domestic Nature": "Nature of domestic incident",
    "Animal Bite Type Description": "Type of animal bite (if applicable)",
    "Poison Nature Description": "Category of poisoning — e.g. '2 Self-harm', '1 Accident', '3 Assault'",
    "Poison Type Description": "Type of poison substance",
    "Poison Description": "Specific poison agent or substance name",
    "Triage Category": "Urgency level: 1 = Critical, 2 = Emergency, 3 = Urgent, 4 = Semi-urgent, 5 = Non-urgent",
    "Mobility Status": "How patient moved around (ambulant, wheelchair, stretcher, etc.)",
    "Conscious Level": "Level of consciousness on arrival",
    "GCS Total Score": "Glasgow Coma Scale total (3-15). Lower = less conscious",
    "Waiting Time (to cubicle)(Minute)": "Minutes waiting to be seen in cubicle",
    "Waiting Time (to triage)(Minute)": "Minutes waiting for triage assessment",
    "Consultation Start Time (Hour, 00-23)": "Hour doctor consultation started",
    "Observation Room Case (Y/N)": "Whether patient was placed in observation room",
    "Observation Room Staying Time (Minute)": "Minutes spent in observation room",
    "Trauma Team Activation (Y/N)": "Whether trauma team was activated",
    "Episode Death (Y/N)": "Whether patient died during this A&E episode",
    "Total Staying Time (Minute)": "Total minutes in A&E from arrival to discharge",
    "Discharge Status (EIS)": "Discharge status code",
    "Discharge Destination (AEIS)": "Where patient went after A&E (e.g. home, admitted, transferred)",
    "A&E to IP Ward: Admission Decision Time (yyyy-mm-dd HH:MM)": "When the decision to admit to inpatient ward was made",
    "A&E to IP Ward: Waiting Time for Admission (Min)": "Minutes waiting for inpatient bed after admission decision",
    "A&E to IP Ward: Length of Stay": "Days in inpatient ward",
    "A&E to IP Ward: Admission Date": "Date admitted to inpatient ward",
    "A&E to IP Ward: Institution": "Hospital where patient was admitted as inpatient",
    "A&E to IP Ward: HN Number": "Hospital Number for the inpatient episode",
    "A&E to IP Ward: Admission Specialty (IPAS)": "Inpatient specialty (e.g. MED, SUR, PSY)",
    "Principal Diagnosis Code": "Primary A&E diagnosis (ICD-9 code)",
    "Diagnosis (rank 2)": "2nd A&E diagnosis code",
    "Diagnosis (rank 3)": "3rd A&E diagnosis code",
    "Diagnosis (rank 4)": "4th A&E diagnosis code",
    "Diagnosis (rank 5)": "5th A&E diagnosis code",
    "Principal Diagnosis Description (HAMDCT)": "Primary A&E diagnosis in plain text",
    "Diagnosis HAMDCT Description (rank 2)": "2nd A&E diagnosis in plain text",
    "Diagnosis HAMDCT Description (rank 3)": "3rd A&E diagnosis in plain text",
    "Diagnosis HAMDCT Description (rank 4)": "4th A&E diagnosis in plain text",
    "Diagnosis HAMDCT Description (rank 5)": "5th A&E diagnosis in plain text",
    "A&E to IP Ward: Principal Diagnosis Code": "Primary inpatient diagnosis (ICD-9 code)",
    "A&E to IP Ward: Diagnosis (rank 2)": "2nd inpatient diagnosis code",
    "A&E to IP Ward: Diagnosis (rank 3)": "3rd inpatient diagnosis code",
    "A&E to IP Ward: Diagnosis (rank 4)": "4th inpatient diagnosis code",
    "A&E to IP Ward: Diagnosis (rank 5)": "5th inpatient diagnosis code",
    "A&E to IP Ward: Principal Diagnosis Description (HAMDCT)": "Primary inpatient diagnosis in plain text",
    "A&E to IP Ward: Diagnosis HAMDCT Description (rank 2)": "2nd inpatient diagnosis in plain text",
    "A&E to IP Ward: Diagnosis HAMDCT Description (rank 3)": "3rd inpatient diagnosis in plain text",
    "A&E to IP Ward: Diagnosis HAMDCT Description (rank 4)": "4th inpatient diagnosis in plain text",
    "A&E to IP Ward: Diagnosis HAMDCT Description (rank 5)": "5th inpatient diagnosis in plain text",
}

# --- Grouped columns ---
COLUMN_GROUPS = {
    "Patient": ["Reference Key", "Sex", "Race Description",
                 "Admission Age (Year) (episode based)"],
    "Attendance": ["Attendance Date (yyyy-mm-dd)", "Attendance Date (Hour)",
                   "Institution (IPAS)", "AE Number", "Attendance Specialty (EIS)",
                   "Admission from OAH (Y/N)", "Ambulance Case (Y/N)"],
    "Triage & Assessment": ["Triage Category", "Mobility Status", "Conscious Level",
                            "GCS Total Score", "Trauma Team Activation (Y/N)"],
    "Diagnosis (A&E)": ["Principal Diagnosis Code", "Diagnosis (rank 2)",
                         "Diagnosis (rank 3)", "Diagnosis (rank 4)", "Diagnosis (rank 5)",
                         "Principal Diagnosis Description (HAMDCT)",
                         "Diagnosis HAMDCT Description (rank 2)",
                         "Diagnosis HAMDCT Description (rank 3)",
                         "Diagnosis HAMDCT Description (rank 4)",
                         "Diagnosis HAMDCT Description (rank 5)"],
    "Diagnosis (Inpatient)": ["A&E to IP Ward: Principal Diagnosis Code",
                               "A&E to IP Ward: Diagnosis (rank 2)",
                               "A&E to IP Ward: Diagnosis (rank 3)",
                               "A&E to IP Ward: Diagnosis (rank 4)",
                               "A&E to IP Ward: Diagnosis (rank 5)",
                               "A&E to IP Ward: Principal Diagnosis Description (HAMDCT)",
                               "A&E to IP Ward: Diagnosis HAMDCT Description (rank 2)",
                               "A&E to IP Ward: Diagnosis HAMDCT Description (rank 3)",
                               "A&E to IP Ward: Diagnosis HAMDCT Description (rank 4)",
                               "A&E to IP Ward: Diagnosis HAMDCT Description (rank 5)"],
    "Poison": ["Poison Nature Description", "Poison Type Description", "Poison Description"],
    "Injury": ["Traumatic Type", "Domestic Type", "Domestic Nature",
               "Animal Bite Type Description"],
    "Discharge": ["Discharge Date (yyyy-mm-dd)", "Discharge Hour (00-23)",
                   "Discharge Status (EIS)", "Discharge Destination (AEIS)",
                   "Total Staying Time (Minute)"],
    "Death": ["Episode Death (Y/N)", "Date of Registered Death", "Exact date of death",
              "Death Cause (Main Cause)", "Death Cause (Supplementary Cause)"],
    "Waiting Times": ["Waiting Time (to cubicle)(Minute)", "Waiting Time (to triage)(Minute)",
                      "Consultation Start Time (Hour, 00-23)",
                      "Observation Room Case (Y/N)", "Observation Room Staying Time (Minute)"],
    "Inpatient": ["A&E to IP Ward: Admission Decision Time (yyyy-mm-dd HH:MM)",
                   "A&E to IP Ward: Waiting Time for Admission (Min)",
                   "A&E to IP Ward: Length of Stay", "A&E to IP Ward: Admission Date",
                   "A&E to IP Ward: Institution", "A&E to IP Ward: HN Number",
                   "A&E to IP Ward: Admission Specialty (IPAS)"],
    "Other": ["Paycode (at discharge)",
              "District of Residence (system code)", "District of Residence Description"],
}

# Build grouped display labels
_grouped_in_order = []
_col_to_display = {}
for group_name, group_cols in COLUMN_GROUPS.items():
    for c in group_cols:
        if c in all_columns:
            display = f"{group_name}  >  {c}"
            _grouped_in_order.append(c)
            _col_to_display[c] = display
for c in all_columns:
    if c not in _col_to_display:
        _grouped_in_order.append(c)
        _col_to_display[c] = c


def _get_suggested_operators(col_name):
    dtype = col_info.get(col_name, "text")
    return OPERATORS_FOR_DTYPE.get(dtype, OPERATORS_FOR_DTYPE["text"])


# Preset prefix for advanced mode
_PRESET_PREFIX = "\u2630 "
unified_column_options = (
    [_PRESET_PREFIX + name for name in COLUMN_PRESETS.keys()]
    + all_columns
)


# =====================================================================
# Shared filter builder — used for both inclusion and exclusion
# =====================================================================

def _render_pick_filters(section_key, label_action="include"):
    """Render a pick-values filter builder.

    section_key: unique prefix for widget keys (e.g. 'incl', 'excl')
    label_action: 'include' or 'exclude' — changes labels like "Select values to include/exclude"

    Returns (pick_filters_list, logic_str).
    """
    filters_key = f"{section_key}_pick_filters"
    logic_key = f"{section_key}_pick_logic"
    editing_key = f"{section_key}_editing_idx"

    if filters_key not in st.session_state:
        st.session_state[filters_key] = []
    if logic_key not in st.session_state:
        st.session_state[logic_key] = "AND"
    if editing_key not in st.session_state:
        st.session_state[editing_key] = None

    pick_filters = st.session_state[filters_key]

    # Logic toggle (only when 2+ filters)
    if len(pick_filters) >= 2:
        if label_action == "include":
            help_text = "**AND** = row must match ALL filters. **OR** = row must match ANY filter."
        else:
            help_text = "**AND** = exclude rows matching ALL rules. **OR** = exclude rows matching ANY rule."
        pick_logic = st.radio(
            "Combine filters with",
            ["AND", "OR"],
            index=0 if st.session_state[logic_key] == "AND" else 1,
            horizontal=True,
            key=f"{section_key}_logic_radio",
            help=help_text,
        )
        st.session_state[logic_key] = pick_logic

    # --- Active filter cards ---
    if pick_filters:
        to_delete = []
        for idx, pf in enumerate(pick_filters):
            with st.container(border=True):
                col_desc = COLUMN_DESCRIPTIONS.get(pf["column"], "")
                header_cols = st.columns([4, 1, 1])
                header_cols[0].markdown(f"**{pf['column']}**")
                if header_cols[1].button("Edit", key=f"{section_key}_edit_{idx}", type="secondary"):
                    st.session_state[editing_key] = idx
                    st.rerun()
                if header_cols[2].button("Remove", key=f"{section_key}_del_{idx}", type="secondary"):
                    to_delete.append(idx)

                if col_desc:
                    st.caption(f"_{col_desc}_")

                n = len(pf["values"])
                preview = ", ".join(str(v) for v in pf["values"][:8])
                if n > 8:
                    preview += f",  ... (+{n - 8} more)"
                st.caption(f"{n} value(s) selected: {preview}")

        if to_delete:
            for i in sorted(to_delete, reverse=True):
                pick_filters.pop(i)
            st.session_state[filters_key] = pick_filters
            st.session_state[editing_key] = None
            st.rerun()

        if len(pick_filters) > 1:
            if st.button("Clear all filters", type="secondary", key=f"{section_key}_clear_all"):
                st.session_state[filters_key] = []
                st.session_state[editing_key] = None
                st.rerun()

    # --- Edit existing filter ---
    editing_idx = st.session_state[editing_key]
    if editing_idx is not None and editing_idx < len(pick_filters):
        pf = pick_filters[editing_idx]
        st.markdown("---")
        st.markdown(f"**Editing: {pf['column']}**")
        desc = COLUMN_DESCRIPTIONS.get(pf["column"], "")
        if desc:
            st.caption(f"_{desc}_")

        _render_value_editor(pf, section_key, "edit", label_action, is_edit=True)

        btn_cols = st.columns(2)
        if btn_cols[1].button("Cancel", key=f"{section_key}_edit_cancel"):
            st.session_state[editing_key] = None
            st.rerun()

    # --- Add new filter ---
    else:
        st.markdown("---")
        st.markdown(f"**Add a filter** — select a column to {label_action} specific values")

        group_names = list(COLUMN_GROUPS.keys())
        pick_cols = st.columns([1, 2])

        with pick_cols[0]:
            group_choice = st.selectbox(
                "Category",
                ["(select a category)"] + group_names,
                key=f"{section_key}_grp_new",
            )

        col_choice = "(select a column)"
        with pick_cols[1]:
            if group_choice != "(select a category)":
                group_cols = [c for c in COLUMN_GROUPS[group_choice] if c in all_columns]
                col_choice = st.selectbox(
                    "Column",
                    ["(select a column)"] + group_cols,
                    key=f"{section_key}_col_new",
                )
            else:
                st.selectbox(
                    "Column",
                    ["(select a category first)"],
                    disabled=True,
                    key=f"{section_key}_col_new_disabled",
                )

        if col_choice != "(select a column)":
            desc = COLUMN_DESCRIPTIONS.get(col_choice, "")
            if desc:
                st.caption(f"_{desc}_")

            # Data preview
            dtype = col_info.get(col_choice, "text")
            with st.expander(f"Data preview: {col_choice}", expanded=False):
                st.caption(f"**Type**: {dtype}")
                vc, total_unique = _get_value_counts(col_choice, top_n=20, mtime=_parquet_mtime)
                st.caption(f"**{total_unique:,}** unique values")
                if dtype == "numeric":
                    stats = _get_numeric_stats(col_choice, mtime=_parquet_mtime)
                    st.caption(
                        f"**Range**: {stats['min']:.1f} – {stats['max']:.1f}  |  "
                        f"**Mean**: {stats['mean']:.1f}  |  "
                        f"**Median**: {stats['median']:.1f}  |  "
                        f"**Missing**: {stats['missing']:,}"
                    )
                preview_data = pd.DataFrame({
                    "Value": vc.index,
                    "Count": vc.values,
                })
                st.dataframe(preview_data, use_container_width=True, hide_index=True, height=250)

            pf_new = {"column": col_choice, "type": _dtype_to_filter_type(dtype), "values": [], "rules": []}
            _render_value_editor(pf_new, section_key, "new", label_action, is_edit=False)

    return pick_filters, st.session_state[logic_key]


def _dtype_to_filter_type(dtype):
    if dtype == "numeric":
        return "numeric"
    if dtype == "date":
        return "date"
    return "text"


def _render_value_editor(pf, section_key, slot, label_action, is_edit):
    """Render the value picker for a filter (new or edit)."""
    filters_key = f"{section_key}_pick_filters"
    editing_key = f"{section_key}_editing_idx"
    pick_filters = st.session_state[filters_key]

    dtype = col_info.get(pf["column"], "text")

    if dtype == "numeric":
        st.caption("Set a range to filter by.")
        existing_rules = pf.get("rules", [])
        cur_min = next((v for op, v in existing_rules if op == ">="), "")
        cur_max = next((v for op, v in existing_rules if op == "<="), "")
        range_cols = st.columns(2)
        min_val = range_cols[0].text_input("Min (>=)", value=cur_min, key=f"{section_key}_{slot}_num_min")
        max_val = range_cols[1].text_input("Max (<=)", value=cur_max, key=f"{section_key}_{slot}_num_max")

        btn_label = "Save" if is_edit else "Add filter"
        if st.button(btn_label, key=f"{section_key}_{slot}_num_btn", type="primary"):
            rules = []
            if min_val.strip():
                rules.append((">=", min_val.strip()))
            if max_val.strip():
                rules.append(("<=", max_val.strip()))
            if rules:
                pf["values"] = [f"{op} {v}" for op, v in rules]
                pf["rules"] = rules
                if not is_edit:
                    pick_filters.append(pf)
                st.session_state[filters_key] = pick_filters
                st.session_state[editing_key] = None
                st.rerun()

    elif dtype == "date":
        st.caption("Set a date range to filter by.")
        existing_rules = pf.get("rules", [])
        cur_after = next((v for op, v in existing_rules if op == "after_date"), "")
        cur_before = next((v for op, v in existing_rules if op == "before_date"), "")
        date_cols = st.columns(2)
        after_val = date_cols[0].text_input("After (yyyy-mm-dd)", value=cur_after, key=f"{section_key}_{slot}_date_after")
        before_val = date_cols[1].text_input("Before (yyyy-mm-dd)", value=cur_before, key=f"{section_key}_{slot}_date_before")

        btn_label = "Save" if is_edit else "Add filter"
        if st.button(btn_label, key=f"{section_key}_{slot}_date_btn", type="primary"):
            rules = []
            if after_val.strip():
                rules.append(("after_date", after_val.strip()))
            if before_val.strip():
                rules.append(("before_date", before_val.strip()))
            if rules:
                pf["values"] = [f"{op} {v}" for op, v in rules]
                pf["rules"] = rules
                if not is_edit:
                    pick_filters.append(pf)
                st.session_state[filters_key] = pick_filters
                st.session_state[editing_key] = None
                st.rerun()

    else:
        with st.spinner(f"Loading values for {pf['column']}..."):
            vc, total_unique = _get_value_counts(pf["column"], mtime=_parquet_mtime)

        if total_unique > 500:
            st.caption(
                f"Showing top 500 of {total_unique:,} unique values. "
                "Use the search box to find specific values, or switch to Advanced mode for regex/ICD-9 ranges."
            )
        else:
            st.caption(f"{total_unique:,} unique values found.")

        options = vc.index.tolist()
        display_map = {v: f"{v}  ({vc[v]:,})" for v in options}

        default_vals = [v for v in pf.get("values", []) if v in options] if is_edit else []

        # Select all / clear buttons
        sel_btn_cols = st.columns([1, 1, 4])
        if sel_btn_cols[0].button("Select all", key=f"{section_key}_{slot}_sel_all", type="secondary"):
            st.session_state[f"{section_key}_{slot}_values_ms"] = options
            st.rerun()
        if sel_btn_cols[1].button("Clear", key=f"{section_key}_{slot}_sel_clear", type="secondary"):
            st.session_state[f"{section_key}_{slot}_values_ms"] = []
            st.rerun()

        selected = st.multiselect(
            f"Select values to {label_action}",
            options=options,
            default=default_vals,
            format_func=lambda x: display_map.get(x, str(x)),
            key=f"{section_key}_{slot}_values_ms",
            placeholder="Search and select values...",
        )

        btn_label = "Save" if is_edit else "Add filter"
        if selected and st.button(btn_label, key=f"{section_key}_{slot}_text_btn", type="primary"):
            pf["values"] = selected
            pf["type"] = "text"
            if not is_edit:
                pick_filters.append(pf)
            st.session_state[filters_key] = pick_filters
            st.session_state[editing_key] = None
            st.rerun()


def _pick_filters_to_rules(pick_filters):
    """Convert pick_filters list to FilterRule list."""
    rules = []
    for pf in pick_filters:
        if pf["type"] == "text":
            rules.append(FilterRule(
                column=pf["column"],
                operator="in_list",
                value=",".join(str(v) for v in pf["values"]),
            ))
        elif pf["type"] == "numeric":
            for op, val in pf.get("rules", []):
                rules.append(FilterRule(column=pf["column"], operator=op, value=val))
        elif pf["type"] == "date":
            for op, val in pf.get("rules", []):
                rules.append(FilterRule(column=pf["column"], operator=op, value=val))
    return rules


def _render_advanced_editor(section_key, rule_group, description):
    """Render the advanced operator-based rule editor."""
    st.caption(description)

    logic = st.radio(
        "Combine rules with",
        ["OR", "AND"],
        index=0 if rule_group.logic == "OR" else 1,
        horizontal=True,
        key=f"{section_key}_adv_logic",
        help="OR = match any rule, AND = match all rules",
    )
    rule_group.logic = logic

    updated_rules = []
    for i, rule in enumerate(rule_group.rules):
        with st.container(border=True):
            if rule.columns:
                current_sel = all_columns[0]
                for preset_name, preset_cols in COLUMN_PRESETS.items():
                    if rule.columns == preset_cols:
                        current_sel = _PRESET_PREFIX + preset_name
                        break
            else:
                current_sel = rule.column or all_columns[0]

            current_idx = unified_column_options.index(current_sel) if current_sel in unified_column_options else 0
            row_cols = st.columns([3, 2, 1])

            with row_cols[0]:
                selected = st.selectbox(
                    "Column / Preset",
                    unified_column_options,
                    index=current_idx,
                    key=f"{section_key}_adv_col_{i}",
                    help="Pick a single column, or a preset group (\u2630) to check multiple columns at once",
                )
                if selected.startswith(_PRESET_PREFIX):
                    preset_name = selected[len(_PRESET_PREFIX):]
                    rule.columns = COLUMN_PRESETS[preset_name]
                    rule.column = ""
                    st.caption(f"{len(rule.columns)} columns")
                else:
                    rule.column = selected
                    rule.columns = []

                # Show description
                effective_col = rule.column or (rule.columns[0] if rule.columns else "")
                desc = COLUMN_DESCRIPTIONS.get(effective_col, "")
                if desc:
                    st.caption(f"_{desc}_")

            with row_cols[1]:
                suggested = _get_suggested_operators(effective_col)
                all_ops = list(OPERATORS.keys())
                ordered_ops = suggested + [op for op in all_ops if op not in suggested]
                op_idx = ordered_ops.index(rule.operator) if rule.operator in ordered_ops else 0
                rule.operator = st.selectbox(
                    "Operator",
                    ordered_ops,
                    index=op_idx,
                    key=f"{section_key}_adv_op_{i}",
                )
                st.caption(OPERATOR_HELP.get(rule.operator, ""))

            with row_cols[2]:
                st.markdown("")
                if st.button("Remove", key=f"{section_key}_adv_del_{i}", type="secondary"):
                    continue

            if rule.operator not in ("is_null", "not_null"):
                rule.value = st.text_input(
                    "Value",
                    value=rule.value,
                    key=f"{section_key}_adv_val_{i}",
                    help=OPERATOR_HELP.get(rule.operator, ""),
                )
                # Inline ICD-9 validation — show error before user runs filter
                if rule.operator == "icd9_range" and rule.value.strip():
                    icd9_err = validate_icd9_range_value(rule.value)
                    if icd9_err:
                        st.error(icd9_err)

            # UIPOL-01: optional step label for STROBE table (exclusion rules only)
            if section_key.startswith("excl"):
                rule.label = st.text_input(
                    "Step label (optional)",
                    value=getattr(rule, "label", ""),
                    key=f"{section_key}_adv_label_{i}",
                    help=(
                        "How this step appears in the STROBE flow table "
                        "(e.g. 'Exclude non-residents'). "
                        "Leave blank to use auto-generated text."
                    ),
                    placeholder="e.g. Exclude non-residents",
                )

            # Data preview for the selected column
            preview_col = rule.column if rule.column else (rule.columns[0] if rule.columns else "")
            if preview_col and preview_col in all_columns:
                with st.expander(f"Preview: {preview_col}", expanded=False):
                    dtype = col_info.get(preview_col, "text")
                    st.caption(f"**Type**: {dtype}")
                    vc, total_unique = _get_value_counts(preview_col, top_n=20, mtime=_parquet_mtime)
                    st.caption(f"**{total_unique:,}** unique values")
                    preview_data = pd.DataFrame({
                        "Value": vc.index,
                        "Count": vc.values,
                    })
                    st.dataframe(preview_data, use_container_width=True, hide_index=True, height=250)

            updated_rules.append(rule)

    rule_group.rules = updated_rules

    if st.button("+ Add rule", key=f"{section_key}_adv_add"):
        rule_group.rules.append(FilterRule())
        st.rerun()

    return rule_group


# =====================================================================
# Inclusion Rules
# =====================================================================
st.subheader("Inclusion Rules")
st.caption(
    "Define which rows to **keep** — only rows matching these criteria are included in the cohort. "
    "All columns are preserved; filtering only removes rows, never columns."
)

incl_mode = st.radio(
    "Mode",
    ["Pick values", "Advanced rules"],
    horizontal=True,
    key="inclusion_mode",
    help="**Pick values**: browse a column and select which values to keep.  \n"
         "**Advanced rules**: use operators like regex, starts_with, ICD-9 ranges, etc.",
)

if incl_mode == "Pick values":
    incl_filters, incl_logic = _render_pick_filters("incl", label_action="include")
    cfg.inclusion = RuleGroup(logic=incl_logic, rules=_pick_filters_to_rules(incl_filters))
else:
    cfg.inclusion = _render_advanced_editor(
        "incl", cfg.inclusion,
        "Use preset groups (\u2630) to search across multiple diagnosis columns at once.",
    )

if not cfg.inclusion.rules:
    st.info("No inclusion filters — all rows will be included.")

st.divider()

# =====================================================================
# Exclusion Rules
# =====================================================================
st.subheader("Exclusion Rules")
st.caption(
    "Define which rows to **remove** from the cohort. "
    "Each rule is applied one at a time so the STROBE flow shows how many are removed at each step."
)

excl_mode = st.radio(
    "Mode",
    ["Pick values", "Advanced rules"],
    horizontal=True,
    key="exclusion_mode",
    help="**Pick values**: browse a column and select which values to exclude.  \n"
         "**Advanced rules**: use operators like regex, starts_with, ICD-9 ranges, etc.",
)

if excl_mode == "Pick values":
    excl_filters, excl_logic = _render_pick_filters("excl", label_action="exclude")
    cfg.exclusion = RuleGroup(logic=excl_logic, rules=_pick_filters_to_rules(excl_filters))
else:
    cfg.exclusion = _render_advanced_editor(
        "excl", cfg.exclusion,
        "Rows matching these rules will be removed from the cohort.",
    )

if not cfg.exclusion.rules:
    st.info("No exclusion filters — no rows will be removed by rule.")

st.markdown("---")
st.markdown("**Special exclusions**")
st.caption("These handle logic the rule engine cannot express.")

# Deaths within buffer
use_death = st.checkbox(
    "Deaths within buffer period",
    value=cfg.death_buffer_days is not None,
    help="Remove presentations where the patient died or had a registered death within N days.",
    key="excl_preset_death",
)
if use_death:
    cfg.death_buffer_days = st.number_input(
        "Death buffer (days)",
        value=cfg.death_buffer_days or 28,
        min_value=0,
        key="death_buffer",
    )
else:
    cfg.death_buffer_days = None

# Incomplete follow-up
use_followup = st.checkbox(
    "Incomplete follow-up",
    value=cfg.followup_buffer_days is not None,
    help="Remove presentations in the last N days of the dataset (insufficient follow-up time).",
    key="excl_preset_followup",
)
if use_followup:
    cfg.followup_buffer_days = st.number_input(
        "Follow-up buffer (days)",
        value=cfg.followup_buffer_days or 28,
        min_value=0,
        key="followup_buffer",
    )
else:
    cfg.followup_buffer_days = None

# First presentation per patient
use_first = st.checkbox(
    "Keep only first presentation per patient",
    value=cfg.first_presentation_only,
    help="After all other exclusions, keep only the earliest attendance per patient.",
    key="excl_first_presentation",
)
cfg.first_presentation_only = use_first

st.session_state["config"] = cfg

# =====================================================================
# Run
# =====================================================================
st.divider()

if st.button("Run Filter", type="primary", use_container_width=True):
    try:
        with st.spinner("Loading master data..."):
            master_df = load_master(use_streamlit_cache=True)

        with st.spinner("Applying filters..."):
            cohort_df, strobe = filter_cohort(master_df, cfg)
            st.session_state["cohort_df"] = cohort_df
            st.session_state["strobe"] = strobe
            st.session_state["master_df"] = master_df

        st.success(f"Cohort extracted: **{len(cohort_df):,}** presentations from **{cohort_df[cfg.patient_id_column].nunique():,}** unique patients")
        st.caption(f"All **{len(cohort_df.columns)}** columns preserved — choose which to export on the Export page.")
    except Exception as exc:
        st.error(
            "Filter failed. This can happen if the master dataset file is corrupt "
            "or a rule value is in an unexpected format. "
            "No cohort results were saved."
        )
        with st.expander("Technical details (for reporting)", expanded=False):
            st.code(str(exc))

# --- STROBE flow ---
if "strobe" in st.session_state:
    strobe = st.session_state["strobe"]

    st.subheader("STROBE Flow")
    st.caption(
        "Each row shows one filtering step applied sequentially. "
        "**Removed** = rows dropped at that step. **Remaining** = rows left after. "
        "Use this table to report cohort derivation in your manuscript (STROBE guideline)."
    )
    flow_df = pd.DataFrame(strobe, columns=["Step", "Removed", "Remaining"])
    st.dataframe(
        flow_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Step": st.column_config.TextColumn(width="large"),
            "Removed": st.column_config.NumberColumn(format="%d"),
            "Remaining": st.column_config.NumberColumn(format="%d"),
        },
    )

    # ERRH-04: Zero-match diagnostics — show sample column values when inclusion filter matched 0 rows
    if "config" in st.session_state:
        zero_match_cols = _detect_zero_match_columns(strobe, st.session_state["config"])
        if zero_match_cols:
            _cfg = st.session_state["config"]
            total_cols = len([
                col
                for rule in _cfg.inclusion.rules
                for col in rule.get_columns()
                if col in col_info
            ])
            st.warning(
                "**No rows matched your inclusion rules.** "
                "Check the sample values below to see what the column actually contains "
                "and compare against the value in your rule."
            )
            if total_cols > 3:
                st.caption(
                    f"Showing 3 of {total_cols} columns targeted by your inclusion rules — "
                    "all were checked during filtering."
                )
            for col in zero_match_cols:
                if col in col_info:
                    with st.expander(f"Sample values in '{col}'", expanded=True):
                        vc, total_unique = _get_value_counts(col, top_n=20, mtime=_parquet_mtime)
                        st.caption(f"{total_unique:,} unique values in master dataset")
                        st.dataframe(
                            pd.DataFrame({"Value": vc.index, "Count": vc.values}),
                            use_container_width=True,
                            hide_index=True,
                        )

    # UIPOL-02: copy-as-text for manuscript Methods section
    with st.expander("Copy STROBE text for Methods section", expanded=False):
        lines = []
        for step, removed, remaining in strobe:
            if removed > 0:
                lines.append(
                    f"{step}: excluded {removed:,} records (n={remaining:,} remaining)"
                )
            else:
                lines.append(f"{step}: n={remaining:,}")
        st.code("\n".join(lines), language=None)
        st.caption(
            "Select all text above and copy (Cmd+A / Ctrl+A, then Cmd+C / Ctrl+C)."
        )

    if "cohort_df" in st.session_state:
        cdf = st.session_state["cohort_df"]
        cfg = st.session_state["config"]

        st.subheader("Quick Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Presentations", f"{len(cdf):,}")
        m2.metric("Unique patients", f"{cdf[cfg.patient_id_column].nunique():,}")

        if "attendance_date" in cdf.columns:
            m3.metric("Date range", f"{cdf['attendance_date'].min().strftime('%Y-%m-%d')} to {cdf['attendance_date'].max().strftime('%Y-%m-%d')}")

        if "Admission Age (Year) (episode based)" in cdf.columns:
            age = pd.to_numeric(cdf["Admission Age (Year) (episode based)"], errors="coerce")
            m4.metric("Mean age", f"{age.mean():.1f}")

        # UIPOL-03: presentations-per-year bar chart
        year_counts = year_bar_chart_data(cdf)
        if year_counts is not None:
            st.markdown("**Presentations per year**")
            st.bar_chart(year_counts.set_index("Year"), use_container_width=True, height=200)

        with st.expander("Preview cohort data", expanded=False):
            preview_cols = [cfg.patient_id_column, "attendance_date", "year"]
            preview_cols += [c for c in ["Admission Age (Year) (episode based)", "Sex",
                            "Principal Diagnosis Code", "Principal Diagnosis Description (HAMDCT)"]
                            if c in cdf.columns]
            preview_cols = [c for c in preview_cols if c in cdf.columns]
            st.dataframe(cdf[preview_cols].head(30), use_container_width=True, hide_index=True)
