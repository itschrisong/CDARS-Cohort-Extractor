"""CohortConfig dataclass — parameterises the flexible rule-based pipeline."""

from __future__ import annotations

import yaml
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Column presets — common multi-column groups in CDARS A&E data
COLUMN_PRESETS = {
    "All diagnosis codes (A&E + IP)": [
        "Principal Diagnosis Code",
        "Diagnosis (rank 2)",
        "Diagnosis (rank 3)",
        "Diagnosis (rank 4)",
        "Diagnosis (rank 5)",
        "A&E to IP Ward: Principal Diagnosis Code",
        "A&E to IP Ward: Diagnosis (rank 2)",
        "A&E to IP Ward: Diagnosis (rank 3)",
        "A&E to IP Ward: Diagnosis (rank 4)",
        "A&E to IP Ward: Diagnosis (rank 5)",
    ],
    "All HAMDCT descriptions": [
        "Principal Diagnosis Description (HAMDCT)",
        "Diagnosis HAMDCT Description (rank 2)",
        "Diagnosis HAMDCT Description (rank 3)",
        "Diagnosis HAMDCT Description (rank 4)",
        "Diagnosis HAMDCT Description (rank 5)",
        "A&E to IP Ward: Principal Diagnosis Description (HAMDCT)",
        "A&E to IP Ward: Diagnosis HAMDCT Description (rank 2)",
        "A&E to IP Ward: Diagnosis HAMDCT Description (rank 3)",
        "A&E to IP Ward: Diagnosis HAMDCT Description (rank 4)",
        "A&E to IP Ward: Diagnosis HAMDCT Description (rank 5)",
    ],
    "All A&E diagnosis codes": [
        "Principal Diagnosis Code",
        "Diagnosis (rank 2)",
        "Diagnosis (rank 3)",
        "Diagnosis (rank 4)",
        "Diagnosis (rank 5)",
    ],
    "All IP diagnosis codes": [
        "A&E to IP Ward: Principal Diagnosis Code",
        "A&E to IP Ward: Diagnosis (rank 2)",
        "A&E to IP Ward: Diagnosis (rank 3)",
        "A&E to IP Ward: Diagnosis (rank 4)",
        "A&E to IP Ward: Diagnosis (rank 5)",
    ],
    "All A&E HAMDCT descriptions": [
        "Principal Diagnosis Description (HAMDCT)",
        "Diagnosis HAMDCT Description (rank 2)",
        "Diagnosis HAMDCT Description (rank 3)",
        "Diagnosis HAMDCT Description (rank 4)",
        "Diagnosis HAMDCT Description (rank 5)",
    ],
    "All IP HAMDCT descriptions": [
        "A&E to IP Ward: Principal Diagnosis Description (HAMDCT)",
        "A&E to IP Ward: Diagnosis HAMDCT Description (rank 2)",
        "A&E to IP Ward: Diagnosis HAMDCT Description (rank 3)",
        "A&E to IP Ward: Diagnosis HAMDCT Description (rank 4)",
        "A&E to IP Ward: Diagnosis HAMDCT Description (rank 5)",
    ],
    "All death-related columns": [
        "Episode Death (Y/N)",
        "Date of Registered Death",
        "Exact date of death",
        "Death Cause (Main Cause)",
        "Death Cause (Supplementary Cause)",
    ],
    "All poison columns": [
        "Poison Nature Description",
        "Poison Type Description",
        "Poison Description",
    ],
}

EXPORT_PRESETS = {
    "Minimal": [
        "Reference Key", "AE Number", "attendance_date", "year", "presentation_id",
    ],
    "Demographics": [
        "Reference Key", "AE Number", "attendance_date", "year", "presentation_id",
        "Sex", "Admission Age (Year) (episode based)", "Race Description",
        "District of Residence Description", "Institution (IPAS)",
    ],
    "Full": None,  # sentinel: all columns
}

# Operator descriptions for UI help text
OPERATOR_HELP = {
    "starts_with": "Text begins with value (e.g. 'E95' matches 'E950', 'E951.2')",
    "contains": "Text contains value anywhere (e.g. 'suicide' matches 'attempt suicide code')",
    "equals": "Exact match after trimming whitespace",
    "not_equals": "Does not match value",
    "in_list": "Matches any item in comma-separated list (e.g. 'CHINA, OVERSEAS, MACAU')",
    "regex": "Regular expression match (e.g. '29[0-6]' matches 290-296)",
    "icd9_range": (
        "ICD-9 numeric range (e.g. '290-319.99, 410-410.99'). "
        "Numeric codes only (001\u2013799). "
        "For E-codes (E800\u2013E999) or V-codes (V01\u2013V99), use starts_with instead "
        "(e.g. 'E95' matches E950\u2013E959)."
    ),
    "before_date": "Date is before value (ISO format: 2020-01-01)",
    "after_date": "Date is after value (ISO format: 2020-01-01)",
    ">": "Numeric greater than",
    "<": "Numeric less than",
    ">=": "Numeric greater than or equal",
    "<=": "Numeric less than or equal",
    "is_null": "Value is missing/empty/NaN (no value needed)",
    "not_null": "Value is present and non-empty (no value needed)",
}

# Suggested operators per column dtype
OPERATORS_FOR_DTYPE = {
    "text": ["starts_with", "contains", "equals", "not_equals", "in_list", "regex", "is_null", "not_null"],
    "numeric": [">", "<", ">=", "<=", "equals", "is_null", "not_null"],
    "date": ["before_date", "after_date", "is_null", "not_null"],
    "diagnosis": ["starts_with", "icd9_range", "regex", "equals", "is_null", "not_null"],
}


@dataclass
class FilterRule:
    """A single filter rule applied to one or more columns."""
    column: str = ""                     # single column name
    columns: List[str] = field(default_factory=list)  # OR: multiple columns (OR logic)
    operator: str = "equals"             # see OPERATORS in cohort.py
    value: str = ""
    case_sensitive: bool = False
    label: str = ""                      # UIPOL-01: optional human-readable STROBE step name

    def get_columns(self) -> List[str]:
        """Return the effective column list (singular or plural)."""
        if self.columns:
            return list(self.columns)
        if self.column:
            return [self.column]
        return []


@dataclass
class RuleGroup:
    """A group of rules combined with AND or OR logic."""
    logic: str = "OR"                    # "AND" or "OR"
    rules: List[FilterRule] = field(default_factory=list)


@dataclass
class CohortConfig:
    """All user-configurable parameters for cohort extraction."""

    # --- Cohort filtering ---
    inclusion: RuleGroup = field(default_factory=RuleGroup)
    exclusion: RuleGroup = field(default_factory=RuleGroup)

    # --- Special clinical exclusions ---
    death_buffer_days: Optional[int] = None
    followup_buffer_days: Optional[int] = None
    first_presentation_only: bool = False

    # --- Column mappings ---
    patient_id_column: str = "Reference Key"
    date_column: str = "Attendance Date (yyyy-mm-dd)"

    eras: Optional[Dict[str, List[int]]] = None

    # --- Feature engineering ---
    features: Dict[str, bool] = field(default_factory=dict)
    lookback_days: int = 365
    recurrence_windows: List[int] = field(default_factory=lambda: [7, 28, 90, 180])
    icd9_flag_definitions: Dict[str, List[List[float]]] = field(default_factory=dict)
    custom_features: List[Dict] = field(default_factory=list)
    custom_lookback_flags: List[Dict] = field(default_factory=list)

    # --- Serialisation ---
    def to_yaml(self, path: str | Path) -> None:
        data = self._to_dict()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_yaml_str(self) -> str:
        return yaml.dump(self._to_dict(), default_flow_style=False, sort_keys=False)

    def _to_dict(self) -> dict:
        """Convert to a YAML-friendly dict."""
        d = {}
        # Inclusion
        d["inclusion"] = self._rule_group_to_dict(self.inclusion)
        # Exclusion
        d["exclusion"] = self._rule_group_to_dict(self.exclusion)
        # Clinical exclusions
        if self.death_buffer_days is not None:
            d["death_buffer_days"] = self.death_buffer_days
        if self.followup_buffer_days is not None:
            d["followup_buffer_days"] = self.followup_buffer_days
        if self.first_presentation_only:
            d["first_presentation_only"] = True
        # Column mappings
        d["patient_id_column"] = self.patient_id_column
        d["date_column"] = self.date_column
        if self.eras:
            d["eras"] = self.eras
        # Feature engineering
        if self.features:
            d["features"] = self.features
        if self.lookback_days != 365:
            d["lookback_days"] = self.lookback_days
        if self.recurrence_windows != [7, 28, 90, 180]:
            d["recurrence_windows"] = self.recurrence_windows
        if self.icd9_flag_definitions:
            d["icd9_flag_definitions"] = self.icd9_flag_definitions
        if self.custom_features:
            d["custom_features"] = self.custom_features
        if self.custom_lookback_flags:
            d["custom_lookback_flags"] = self.custom_lookback_flags
        return d

    @staticmethod
    def _rule_group_to_dict(rg: RuleGroup) -> dict:
        rules = []
        for r in rg.rules:
            rd = {}
            if r.columns:
                rd["columns"] = list(r.columns)
            elif r.column:
                rd["column"] = r.column
            rd["operator"] = r.operator
            if r.operator not in ("is_null", "not_null"):
                rd["value"] = r.value
            if r.case_sensitive:
                rd["case_sensitive"] = True
            if r.label:
                rd["label"] = r.label
            rules.append(rd)
        return {"logic": rg.logic, "rules": rules}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CohortConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_yaml_str(cls, text: str) -> "CohortConfig":
        data = yaml.safe_load(text)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "CohortConfig":
        inclusion = cls._parse_rule_group(data.get("inclusion", {}))
        exclusion = cls._parse_rule_group(data.get("exclusion", {}))

        # Parse icd9_flag_definitions — normalize to list of [float, float] pairs
        raw_icd9 = data.get("icd9_flag_definitions", {})
        icd9_defs = {}
        if raw_icd9:
            for name, ranges in raw_icd9.items():
                icd9_defs[name] = [[float(r[0]), float(r[1])] for r in ranges]

        return cls(
            inclusion=inclusion,
            exclusion=exclusion,
            death_buffer_days=data.get("death_buffer_days"),
            followup_buffer_days=data.get("followup_buffer_days"),
            first_presentation_only=data.get("first_presentation_only", False),
            patient_id_column=data.get("patient_id_column", "Reference Key"),
            date_column=data.get("date_column", "Attendance Date (yyyy-mm-dd)"),
            eras=data.get("eras"),
            features=data.get("features", {}),
            lookback_days=data.get("lookback_days", 365),
            recurrence_windows=data.get("recurrence_windows", [7, 28, 90, 180]),
            icd9_flag_definitions=icd9_defs,
            custom_features=data.get("custom_features", []),
            custom_lookback_flags=data.get("custom_lookback_flags", []),
        )

    @classmethod
    def _parse_rule_group(cls, data: dict) -> RuleGroup:
        if not data:
            return RuleGroup()
        rules = []
        for rd in data.get("rules", []):
            rule = FilterRule(
                column=rd.get("column", ""),
                columns=rd.get("columns", []),
                operator=rd.get("operator", "equals"),
                value=str(rd.get("value", "")),
                case_sensitive=rd.get("case_sensitive", False),
                label=rd.get("label", ""),
            )
            rules.append(rule)
        return RuleGroup(logic=data.get("logic", "OR"), rules=rules)

    def config_hash(self) -> str:
        """Return a short hash for caching keys."""
        return hashlib.md5(
            self.to_yaml_str().encode()
        ).hexdigest()[:12]
