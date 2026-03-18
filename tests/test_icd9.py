"""Tests for core.icd9 module."""

import pandas as pd
import pytest

from core.icd9 import flag_codes, has_icd9_range, compute_charlson


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "dx1": ["E950", "296.2", "250.0", "999", "nan"],
            "dx2": ["428", "E959.1", "571.5", "301.0", "nan"],
        }
    )


class TestFlagCodes:
    def test_e95_prefix(self, sample_df):
        result = flag_codes(sample_df, ["dx1", "dx2"], "E95")
        assert result.tolist() == [True, True, False, False, False]

    def test_no_match(self, sample_df):
        result = flag_codes(sample_df, ["dx1", "dx2"], "Z99")
        assert not result.any()

    def test_missing_column(self, sample_df):
        result = flag_codes(sample_df, ["dx1", "nonexistent"], "E95")
        assert result[0] == True

    def test_single_prefix_char(self, sample_df):
        result = flag_codes(sample_df, ["dx1"], "2")
        assert result[1] == True  # 296.2
        assert result[2] == True  # 250.0


class TestHasIcd9Range:
    def test_mental_disorders(self, sample_df):
        result = has_icd9_range(sample_df, ["dx1", "dx2"], [(290, 319.99)])
        # dx1: 296.2 → match; dx2: 301.0 → match
        assert result[1] == 1  # 296.2
        assert result[3] == 1  # 301.0

    def test_chf(self, sample_df):
        result = has_icd9_range(sample_df, ["dx2"], [(428, 428.99)])
        assert result[0] == 1  # 428
        assert result[1] == 0

    def test_ecodes_ignored(self, sample_df):
        # E-codes are non-numeric, should be ignored
        result = has_icd9_range(sample_df, ["dx1"], [(950, 959)])
        assert result[0] == 0  # E950 is not numeric 950


class TestComputeCharlson:
    def test_no_conditions(self):
        assert compute_charlson(["999", "888"]) == 0

    def test_single_condition(self):
        # CHF (428.x) → weight 1
        assert compute_charlson(["428.0"]) == 1

    def test_multiple_conditions(self):
        # CHF (1) + CVD (1) = 2
        assert compute_charlson(["428.0", "430"]) == 2

    def test_hierarchy_dm(self):
        # DM uncomplicated (1) + DM complicated (2) → only 2
        assert compute_charlson(["250.0", "250.5"]) == 2

    def test_hierarchy_liver(self):
        # Liver mild (1) + Liver severe (3) → only 3
        assert compute_charlson(["571.2", "572.2"]) == 3

    def test_hierarchy_cancer(self):
        # Cancer (2) + Metastatic (6) → only 6
        assert compute_charlson(["150", "196"]) == 6

    def test_ecodes_skipped(self):
        assert compute_charlson(["E950", "E980"]) == 0

    def test_empty(self):
        assert compute_charlson([]) == 0

    def test_non_numeric(self):
        assert compute_charlson(["nan", "None", ""]) == 0
