"""Shared whitespace and Unicode normalization for CDARS data.

CDARS data has inconsistent spacing/NBSP and Unicode hyphens.
These helpers are used by both the cohort filter engine and the ingestion pipeline.
"""

import re

# Unicode dashes/hyphens normalized to ASCII hyphen-minus (U+002D)
_UNICODE_HYPHENS_RE = r'[\u2010\u2011\u2012\u2013\u2014\u2015]'


def normalize_ws_series(series):
    """Normalize Unicode spaces, hyphens, and whitespace in a pandas string Series.

    Steps:
    1. Unicode hyphens/dashes (U+2010-U+2015) -> ASCII hyphen-minus
    2. Zero-width space (U+200B) -> removed
    3. Remaining whitespace (\\s covers U+2009 thin space) + NBSP (\\xa0) -> single space
    """
    result = series.str.replace(_UNICODE_HYPHENS_RE, '-', regex=True)
    result = result.str.replace('\u200b', '', regex=False)
    return result.str.replace(r'[\s\xa0]+', ' ', regex=True).str.strip()


def normalize_ws_scalar(v: str) -> str:
    """Scalar equivalent of normalize_ws_series() for single string values."""
    v = re.sub(_UNICODE_HYPHENS_RE, '-', v)
    v = v.replace('\u200b', '')
    return re.sub(r'[\s\xa0]+', ' ', v).strip()
