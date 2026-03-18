"""Tests for the generic rule-based cohort engine."""

import pandas as pd
import numpy as np
import pytest

from core.config import CohortConfig, FilterRule, RuleGroup, COLUMN_PRESETS
from core.cohort import apply_rule, apply_rule_group, filter_cohort, OPERATORS


def _make_master(n=100):
    """Create a synthetic master DataFrame."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=365, freq="D")

    df = pd.DataFrame(
        {
            "Reference Key": [f"P{i:04d}" for i in rng.integers(0, 50, n)],
            "AE Number": [f"AE{i}" for i in range(n)],
            "Attendance Date (yyyy-mm-dd)": rng.choice(dates, n),
            "Admission Age (Year) (episode based)": rng.integers(10, 80, n).astype(float),
            "Sex": rng.choice(["M", "F"], n),
            "Triage Category": rng.choice([1, 2, 3, 4, 5], n).astype(float),
            "Principal Diagnosis Code": ["999"] * n,
            "Diagnosis (rank 2)": ["nan"] * n,
            "Diagnosis (rank 3)": ["nan"] * n,
            "Diagnosis (rank 4)": ["nan"] * n,
            "Diagnosis (rank 5)": ["nan"] * n,
            "A&E to IP Ward: Principal Diagnosis Code": ["nan"] * n,
            "A&E to IP Ward: Diagnosis (rank 2)": ["nan"] * n,
            "A&E to IP Ward: Diagnosis (rank 3)": ["nan"] * n,
            "A&E to IP Ward: Diagnosis (rank 4)": ["nan"] * n,
            "A&E to IP Ward: Diagnosis (rank 5)": ["nan"] * n,
            "Principal Diagnosis Description (HAMDCT)": ["other"] * n,
            "Diagnosis HAMDCT Description (rank 2)": ["nan"] * n,
            "Poison Nature Description": ["nan"] * n,
            "Paycode (at discharge)": ["EP1"] * n,
            "District of Residence Description": ["LOCAL"] * n,
            "Episode Death (Y/N)": ["N"] * n,
            "Date of Registered Death": [pd.NaT] * n,
        }
    )
    # Inject some self-harm E-codes
    df.loc[0:9, "Diagnosis (rank 2)"] = "E950"
    df.loc[10:14, "Diagnosis (rank 3)"] = "E951.2"
    # Inject some minors
    df.loc[0:2, "Admission Age (Year) (episode based)"] = 15.0
    return df


# ---- Operator tests ----

class TestOperators:
    def test_starts_with(self):
        df = _make_master()
        rule = FilterRule(column="Diagnosis (rank 2)", operator="starts_with", value="E95")
        mask = apply_rule(df, rule)
        assert mask.sum() == 10  # rows 0-9

    def test_starts_with_case_insensitive(self):
        df = pd.DataFrame({"col": ["Apple", "APPLE", "banana"]})
        rule = FilterRule(column="col", operator="starts_with", value="apple", case_sensitive=False)
        assert apply_rule(df, rule).sum() == 2

    def test_contains(self):
        df = pd.DataFrame({"desc": ["attempt suicide code", "no match", "ATTEMPT SUICIDE"]})
        rule = FilterRule(column="desc", operator="contains", value="attempt suicide")
        assert apply_rule(df, rule).sum() == 2  # case insensitive by default

    def test_equals(self):
        df = pd.DataFrame({"status": ["Y", "N", " Y ", "y"]})
        rule = FilterRule(column="status", operator="equals", value="Y")
        # strips whitespace, case insensitive by default
        assert apply_rule(df, rule).sum() == 3

    def test_not_equals(self):
        df = pd.DataFrame({"status": ["Y", "N", "N"]})
        rule = FilterRule(column="status", operator="not_equals", value="Y")
        assert apply_rule(df, rule).sum() == 2

    def test_in_list(self):
        df = pd.DataFrame({"dist": ["CHINA", "LOCAL", "OVERSEAS", "MACAU", "LOCAL"]})
        rule = FilterRule(column="dist", operator="in_list", value="CHINA, OVERSEAS, MACAU")
        assert apply_rule(df, rule).sum() == 3

    def test_regex(self):
        df = pd.DataFrame({"code": ["E950", "E951", "E960", "428"]})
        rule = FilterRule(column="code", operator="regex", value=r"e95[0-9]")
        assert apply_rule(df, rule).sum() == 2  # case insensitive

    def test_before_date(self):
        df = pd.DataFrame({"date": ["2020-06-01", "2020-12-01", "2021-01-15"]})
        rule = FilterRule(column="date", operator="before_date", value="2021-01-01")
        assert apply_rule(df, rule).sum() == 2

    def test_after_date(self):
        df = pd.DataFrame({"date": ["2020-06-01", "2020-12-01", "2021-01-15"]})
        rule = FilterRule(column="date", operator="after_date", value="2020-10-01")
        assert apply_rule(df, rule).sum() == 2

    def test_gt(self):
        df = pd.DataFrame({"age": [15, 18, 25, 60]})
        rule = FilterRule(column="age", operator=">", value="18")
        assert apply_rule(df, rule).sum() == 2

    def test_lt(self):
        df = pd.DataFrame({"age": [15, 18, 25, 60]})
        rule = FilterRule(column="age", operator="<", value="18")
        assert apply_rule(df, rule).sum() == 1

    def test_gte(self):
        df = pd.DataFrame({"age": [15, 18, 25]})
        rule = FilterRule(column="age", operator=">=", value="18")
        assert apply_rule(df, rule).sum() == 2

    def test_lte(self):
        df = pd.DataFrame({"age": [15, 18, 25]})
        rule = FilterRule(column="age", operator="<=", value="18")
        assert apply_rule(df, rule).sum() == 2

    def test_is_null(self):
        df = pd.DataFrame({"val": [None, "", "nan", "hello", "None"]})
        rule = FilterRule(column="val", operator="is_null")
        assert apply_rule(df, rule).sum() == 4

    def test_not_null(self):
        df = pd.DataFrame({"val": [None, "", "nan", "hello", "None"]})
        rule = FilterRule(column="val", operator="not_null")
        assert apply_rule(df, rule).sum() == 1

    def test_icd9_range(self):
        df = pd.DataFrame({
            "Principal Diagnosis Code": ["296.2", "428", "999", "311"],
            "Diagnosis (rank 2)": ["nan", "nan", "295.1", "nan"],
        })
        rule = FilterRule(
            columns=["Principal Diagnosis Code", "Diagnosis (rank 2)"],
            operator="icd9_range",
            value="290-319.99",
        )
        mask = apply_rule(df, rule)
        assert mask.sum() == 3  # 296.2, 311, 295.1


# ---- Multi-column OR tests ----

class TestMultiColumn:
    def test_multi_column_or(self):
        df = _make_master()
        # E-codes are in rank 2 (rows 0-9) and rank 3 (rows 10-14)
        rule = FilterRule(
            columns=["Diagnosis (rank 2)", "Diagnosis (rank 3)"],
            operator="starts_with",
            value="E95",
        )
        mask = apply_rule(df, rule)
        assert mask.sum() == 15

    def test_missing_columns_ignored(self):
        df = pd.DataFrame({"col_a": ["hello", "world"]})
        rule = FilterRule(
            columns=["col_a", "nonexistent_col"],
            operator="equals",
            value="hello",
        )
        assert apply_rule(df, rule).sum() == 1


# ---- RuleGroup AND/OR tests ----

class TestRuleGroup:
    def test_or_logic(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["1", "2", "3"]})
        group = RuleGroup(logic="OR", rules=[
            FilterRule(column="a", operator="equals", value="x"),
            FilterRule(column="b", operator="equals", value="3"),
        ])
        assert apply_rule_group(df, group).sum() == 2

    def test_and_logic(self):
        df = pd.DataFrame({"a": ["x", "x", "y"], "b": ["1", "2", "1"]})
        group = RuleGroup(logic="AND", rules=[
            FilterRule(column="a", operator="equals", value="x"),
            FilterRule(column="b", operator="equals", value="1"),
        ])
        assert apply_rule_group(df, group).sum() == 1

    def test_empty_group_returns_false(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        group = RuleGroup()
        assert apply_rule_group(df, group).sum() == 0


# ---- filter_cohort integration tests ----

class TestFilterCohort:
    def test_inclusion_only(self):
        df = _make_master()
        cfg = CohortConfig(
            inclusion=RuleGroup(logic="OR", rules=[
                FilterRule(
                    columns=["Diagnosis (rank 2)", "Diagnosis (rank 3)"],
                    operator="starts_with",
                    value="E95",
                ),
            ]),
        )
        cohort, strobe = filter_cohort(df, cfg)
        assert len(cohort) == 15
        assert "presentation_id" in cohort.columns

    def test_inclusion_plus_exclusion(self):
        df = _make_master()
        cfg = CohortConfig(
            inclusion=RuleGroup(logic="OR", rules=[
                FilterRule(
                    columns=["Diagnosis (rank 2)", "Diagnosis (rank 3)"],
                    operator="starts_with",
                    value="E95",
                ),
            ]),
            exclusion=RuleGroup(logic="OR", rules=[
                FilterRule(
                    column="Admission Age (Year) (episode based)",
                    operator="<",
                    value="18",
                ),
            ]),
        )
        cohort, strobe = filter_cohort(df, cfg)
        # Rows 0-2 are age 15 and have E-codes → excluded
        assert len(cohort) == 12

    def test_strobe_flow_structure(self):
        df = _make_master()
        cfg = CohortConfig(
            inclusion=RuleGroup(logic="OR", rules=[
                FilterRule(column="Diagnosis (rank 2)", operator="starts_with", value="E95"),
            ]),
            exclusion=RuleGroup(logic="OR", rules=[
                FilterRule(column="Admission Age (Year) (episode based)", operator="<", value="18"),
                FilterRule(column="District of Residence Description", operator="in_list", value="CHINA, OVERSEAS"),
            ]),
        )
        cohort, strobe = filter_cohort(df, cfg)
        # strobe is list of (label, removed, remaining)
        assert len(strobe) >= 4  # master, inclusion, 2 exclusions, analysis
        assert strobe[0][0] == "Total records in master"
        assert strobe[-1][0] == "Analysis cohort"

    def test_death_buffer(self):
        df = _make_master(20)
        # Set all rows to have E-code
        df["Diagnosis (rank 2)"] = "E950"
        df.loc[0, "Episode Death (Y/N)"] = "Y"
        cfg = CohortConfig(
            inclusion=RuleGroup(logic="OR", rules=[
                FilterRule(column="Diagnosis (rank 2)", operator="starts_with", value="E95"),
            ]),
            death_buffer_days=28,
        )
        cohort, strobe = filter_cohort(df, cfg)
        assert len(cohort) < 20

    def test_no_inclusion_returns_all(self):
        df = _make_master(10)
        cfg = CohortConfig()
        cohort, strobe = filter_cohort(df, cfg)
        assert len(cohort) == 10

    def test_presentation_id_unique(self):
        df = _make_master()
        cfg = CohortConfig(
            inclusion=RuleGroup(logic="OR", rules=[
                FilterRule(column="Diagnosis (rank 2)", operator="starts_with", value="E95"),
            ]),
        )
        cohort, _ = filter_cohort(df, cfg)
        assert cohort["presentation_id"].is_unique

    def test_no_mutation_of_input(self):
        """filter_cohort must NOT mutate the caller's DataFrame (FNDN-03 CoW safety).

        The function adds 'presentation_id' to the cohort; it must not add it
        (or any other column) to the original df passed in.
        """
        df = _make_master(20)
        cols_before = set(df.columns)

        cfg = CohortConfig(
            inclusion=RuleGroup(logic="OR", rules=[
                FilterRule(column="Diagnosis (rank 2)", operator="starts_with", value="E95"),
            ]),
        )
        filter_cohort(df, cfg)

        cols_after = set(df.columns)
        assert cols_before == cols_after, (
            f"filter_cohort mutated the input DataFrame. "
            f"Added columns: {cols_after - cols_before}"
        )


# ---- Config YAML round-trip ----

class TestConfigSerde:
    def test_round_trip(self, tmp_path):
        cfg = CohortConfig(
            inclusion=RuleGroup(logic="OR", rules=[
                FilterRule(column="code", operator="starts_with", value="E95"),
                FilterRule(columns=["col1", "col2"], operator="contains", value="test"),
            ]),
            exclusion=RuleGroup(logic="OR", rules=[
                FilterRule(column="age", operator="<", value="18"),
            ]),
            death_buffer_days=28,
            eras={"A": [2020, 2021]},
        )
        path = tmp_path / "test.yaml"
        cfg.to_yaml(path)
        loaded = CohortConfig.from_yaml(path)

        assert len(loaded.inclusion.rules) == 2
        assert loaded.inclusion.rules[0].operator == "starts_with"
        assert loaded.inclusion.rules[1].columns == ["col1", "col2"]
        assert len(loaded.exclusion.rules) == 1
        assert loaded.death_buffer_days == 28
        assert loaded.eras == {"A": [2020, 2021]}

    def test_column_presets_exist(self):
        assert "All diagnosis codes (A&E + IP)" in COLUMN_PRESETS
        assert len(COLUMN_PRESETS["All diagnosis codes (A&E + IP)"]) == 10


# ---- Whitespace normalization tests (CDARS data quality) ----

class TestWhitespaceNormalization:
    """CDARS data has inconsistent whitespace — e.g. '2 Self-harm' vs '2  Self-harm'."""

    def test_contains_double_space(self):
        df = pd.DataFrame({"desc": ["2 Self-harm", "2  Self-harm", "4 Others"]})
        rule = FilterRule(column="desc", operator="contains", value="Self-harm")
        assert apply_rule(df, rule).sum() == 2

    def test_starts_with_double_space(self):
        df = pd.DataFrame({"desc": ["2 Self-harm", "2  Self-harm", "4 Others"]})
        rule = FilterRule(column="desc", operator="starts_with", value="2 Self")
        assert apply_rule(df, rule).sum() == 2

    def test_equals_double_space(self):
        df = pd.DataFrame({"desc": ["2 Self-harm", "2  Self-harm"]})
        rule = FilterRule(column="desc", operator="equals", value="2 Self-harm")
        assert apply_rule(df, rule).sum() == 2

    def test_in_list_double_space(self):
        df = pd.DataFrame({"desc": ["2 Self-harm", "2  Self-harm", "4 Others"]})
        rule = FilterRule(column="desc", operator="in_list", value="2 Self-harm, 4 Others")
        assert apply_rule(df, rule).sum() == 3

    def test_nbsp_normalized(self):
        """Non-breaking spaces (\\xa0) should be treated as regular spaces."""
        df = pd.DataFrame({"desc": ["1\xa0\xa0Accident", "1 Accident"]})
        rule = FilterRule(column="desc", operator="equals", value="1 Accident")
        assert apply_rule(df, rule).sum() == 2

    def test_is_null_with_whitespace_only(self):
        df = pd.DataFrame({"val": [None, "", "  ", "nan", "hello"]})
        rule = FilterRule(column="val", operator="is_null")
        mask = apply_rule(df, rule)
        # None, empty, whitespace-only, and "nan" are all null
        assert mask.sum() == 4

    # ---- FNDN-04: Unicode character normalization (RED — not yet implemented) ----

    def test_zwsp_normalized(self):
        """Zero-width space (U+200B) in data must be stripped before matching."""
        # "2 \u200bSelf-harm" has space+ZWSP; after stripping ZWSP and collapsing
        # whitespace it should match "2 Self-harm"
        df = pd.DataFrame({"desc": ["2 \u200bSelf-harm", "4 Others"]})
        rule = FilterRule(column="desc", operator="starts_with", value="2 Self-harm")
        assert apply_rule(df, rule).sum() == 1, (
            "ZWSP in data should be stripped so '2 \\u200bSelf-harm' matches '2 Self-harm'"
        )

    def test_unicode_hyphen_normalized(self):
        """En-dash (U+2013) in data must be normalized to ASCII hyphen '-'."""
        # "Self\u2013harm" uses an en-dash; the rule uses an ASCII hyphen
        df = pd.DataFrame({"desc": ["Self\u2013harm", "Other"]})
        rule = FilterRule(column="desc", operator="contains", value="Self-harm")
        assert apply_rule(df, rule).sum() == 1, (
            "En-dash U+2013 in data should normalize to '-' and match 'Self-harm'"
        )

    def test_zwsp_in_value(self):
        """Zero-width space (U+200B) in the *rule value* must be stripped too."""
        # Data is clean; the ZWSP is in what the user typed as the match value
        df = pd.DataFrame({"desc": ["2 Self-harm", "4 Others"]})
        rule = FilterRule(column="desc", operator="starts_with", value="2 \u200bSelf-harm")
        assert apply_rule(df, rule).sum() == 1, (
            "ZWSP in rule value should be stripped so '2 \\u200bSelf-harm' matches '2 Self-harm'"
        )


# ---- FilterRule label field and STROBE text format (UIPOL-01/02) ----

class TestFilterLabel:
    """RED tests for FilterRule.label field and STROBE label behaviour.

    These tests FAIL until plan 04-02 adds label to FilterRule and
    updates filter_cohort() to use it.
    """

    def _make_df(self):
        """Minimal DataFrame for STROBE integration tests."""
        return pd.DataFrame(
            {
                "Reference Key": [f"P{i:03d}" for i in range(10)],
                "AE Number": [f"AE{i}" for i in range(10)],
                "Attendance Date (yyyy-mm-dd)": pd.date_range("2020-01-01", periods=10),
                "Admission Age (Year) (episode based)": [20.0] * 7 + [15.0] * 3,
                "Sex": ["M"] * 10,
                "Triage Category": [3.0] * 10,
                "Principal Diagnosis Code": ["999"] * 10,
                "Diagnosis (rank 2)": ["E950"] * 10,
                "Diagnosis (rank 3)": ["nan"] * 10,
                "Diagnosis (rank 4)": ["nan"] * 10,
                "Diagnosis (rank 5)": ["nan"] * 10,
                "A&E to IP Ward: Principal Diagnosis Code": ["nan"] * 10,
                "A&E to IP Ward: Diagnosis (rank 2)": ["nan"] * 10,
                "A&E to IP Ward: Diagnosis (rank 3)": ["nan"] * 10,
                "A&E to IP Ward: Diagnosis (rank 4)": ["nan"] * 10,
                "A&E to IP Ward: Diagnosis (rank 5)": ["nan"] * 10,
                "Principal Diagnosis Description (HAMDCT)": ["other"] * 10,
                "Diagnosis HAMDCT Description (rank 2)": ["nan"] * 10,
                "Poison Nature Description": ["nan"] * 10,
                "Paycode (at discharge)": ["EP1"] * 10,
                "District of Residence Description": ["LOCAL"] * 10,
                "Episode Death (Y/N)": ["N"] * 10,
                "Date of Registered Death": [pd.NaT] * 10,
            }
        )

    def test_label_round_trip(self):
        """FilterRule.label survives a YAML round-trip.

        FAILS until FilterRule gains a label field (plan 04-02).
        """
        cfg = CohortConfig(
            exclusion=RuleGroup(logic="OR", rules=[
                FilterRule(
                    column="District of Residence Description",
                    operator="in_list",
                    value="CHINA, OVERSEAS",
                    label="Exclude non-residents",
                ),
            ]),
        )
        yaml_str = cfg.to_yaml_str()
        loaded = CohortConfig.from_yaml_str(yaml_str)
        assert loaded.exclusion.rules[0].label == "Exclude non-residents", (
            "FilterRule.label did not survive YAML round-trip"
        )

    def test_strobe_label_used(self):
        """filter_cohort() uses rule.label as the STROBE step string when set.

        FAILS until filter_cohort() reads rule.label (plan 04-02).
        """
        df = self._make_df()
        cfg = CohortConfig(
            inclusion=RuleGroup(logic="OR", rules=[
                FilterRule(column="Diagnosis (rank 2)", operator="starts_with", value="E95"),
            ]),
            exclusion=RuleGroup(logic="OR", rules=[
                FilterRule(
                    column="Admission Age (Year) (episode based)",
                    operator="<",
                    value="18",
                    label="My custom label",
                ),
            ]),
        )
        _, strobe = filter_cohort(df, cfg)
        step_labels = [step[0] for step in strobe]
        assert "My custom label" in step_labels, (
            f"Expected 'My custom label' in STROBE steps, got: {step_labels}"
        )

    def test_strobe_label_fallback(self):
        """filter_cohort() falls back to auto-generated 'Excl:' prefix when rule.label is empty.

        FAILS until filter_cohort() reads rule.label (plan 04-02).
        """
        df = self._make_df()
        cfg = CohortConfig(
            inclusion=RuleGroup(logic="OR", rules=[
                FilterRule(column="Diagnosis (rank 2)", operator="starts_with", value="E95"),
            ]),
            exclusion=RuleGroup(logic="OR", rules=[
                FilterRule(
                    column="Admission Age (Year) (episode based)",
                    operator="<",
                    value="18",
                    label="",
                ),
            ]),
        )
        _, strobe = filter_cohort(df, cfg)
        excl_steps = [step[0] for step in strobe if step[0] not in ("Total records in master", "After inclusion filter", "Analysis cohort")]
        assert len(excl_steps) >= 1, "Expected at least one exclusion STROBE step"
        assert excl_steps[0].startswith("Excl:"), (
            f"Expected exclusion step to start with 'Excl:', got: {excl_steps[0]!r}"
        )

    def test_strobe_text_format(self):
        """STROBE list formats into manuscript-ready text lines (UIPOL-02).

        This test is self-contained — it defines the formatting contract inline
        without importing any page module.  It will PASS immediately as a pure
        specification test; the Wave 2 implementation must match this format.
        """

        def _format_strobe_text(strobe: list) -> str:
            """Expected manuscript format for STROBE output.

            strobe items: (label, removed, remaining)
            Initial step (removed==0): "Step: n=R"
            Other steps with removed>0: "Step: excluded N records (n=R remaining)"
            """
            lines = []
            for label, removed, remaining in strobe:
                if removed == 0:
                    lines.append(f"{label}: n={remaining}")
                else:
                    lines.append(f"{label}: excluded {removed} records (n={remaining} remaining)")
            return "\n".join(lines)

        strobe = [
            ("Total records in master", 0, 100),
            ("Meets inclusion criteria", 40, 60),
            ("Excl: age < 18", 10, 50),
            ("Analysis cohort", 0, 50),
        ]
        result = _format_strobe_text(strobe)
        expected = (
            "Total records in master: n=100\n"
            "Meets inclusion criteria: excluded 40 records (n=60 remaining)\n"
            "Excl: age < 18: excluded 10 records (n=50 remaining)\n"
            "Analysis cohort: n=50"
        )
        assert result == expected, (
            f"STROBE text format mismatch.\nGot:\n{result}\nExpected:\n{expected}"
        )


# ---- ICD-9 range input validation (ERRH-03) ----

class TestIcd9Validation:
    """Tests for validate_icd9_range_value() and hardened apply_rule() icd9_range branch."""

    def test_e_code_upper_returns_error(self):
        """E-codes must return a non-None error string mentioning starts_with."""
        from core.cohort import validate_icd9_range_value
        err = validate_icd9_range_value("E950-E959")
        assert err is not None, "Expected error for E-code range, got None"
        assert "starts_with" in err, f"Error should mention starts_with, got: {err!r}"

    def test_v_code_upper_returns_error(self):
        """V-codes must return a non-None error string."""
        from core.cohort import validate_icd9_range_value
        err = validate_icd9_range_value("V01-V05")
        assert err is not None, "Expected error for V-code range, got None"
        assert "starts_with" in err, f"Error should mention starts_with, got: {err!r}"

    def test_e_code_lower_returns_error(self):
        """Lowercase e-codes are also invalid (case-insensitive check)."""
        from core.cohort import validate_icd9_range_value
        err = validate_icd9_range_value("e800-e899")
        assert err is not None, "Expected error for lowercase e-code range, got None"

    def test_v_code_lower_returns_error(self):
        """Lowercase v-codes are also invalid (case-insensitive check)."""
        from core.cohort import validate_icd9_range_value
        err = validate_icd9_range_value("v10-v19")
        assert err is not None, "Expected error for lowercase v-code range, got None"

    def test_valid_single_range_returns_none(self):
        """Valid numeric range '290-319.99' returns None (no error)."""
        from core.cohort import validate_icd9_range_value
        err = validate_icd9_range_value("290-319.99")
        assert err is None, f"Expected None for valid range, got: {err!r}"

    def test_valid_multi_range_returns_none(self):
        """Valid multi-range '290-319.99, 410-410.99' returns None."""
        from core.cohort import validate_icd9_range_value
        err = validate_icd9_range_value("290-319.99, 410-410.99")
        assert err is None, f"Expected None for valid multi-range, got: {err!r}"

    def test_non_numeric_bounds_returns_error(self):
        """Non-numeric bounds like '290-abc' return an error string."""
        from core.cohort import validate_icd9_range_value
        err = validate_icd9_range_value("290-abc")
        assert err is not None, "Expected error for non-numeric bounds, got None"

    def test_missing_hyphen_returns_error(self):
        """A value without a hyphen (e.g. '290') is not a valid range."""
        from core.cohort import validate_icd9_range_value
        err = validate_icd9_range_value("290")
        assert err is not None, "Expected error for single number without hyphen, got None"

    def test_empty_string_returns_none(self):
        """Empty string returns None — user may not have typed yet."""
        from core.cohort import validate_icd9_range_value
        err = validate_icd9_range_value("")
        assert err is None, f"Empty string should return None, got: {err!r}"

    def test_apply_rule_e_code_raises_value_error(self):
        """apply_rule() with icd9_range + E-code value raises ValueError with starts_with message."""
        df = pd.DataFrame({
            "Principal Diagnosis Code": ["E950", "E951", "999"],
        })
        rule = FilterRule(
            column="Principal Diagnosis Code",
            operator="icd9_range",
            value="E950-E959",
        )
        with pytest.raises(ValueError) as exc_info:
            apply_rule(df, rule)
        assert "starts_with" in str(exc_info.value), (
            f"ValueError should mention starts_with, got: {exc_info.value!r}"
        )
