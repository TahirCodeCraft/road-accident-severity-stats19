"""F0 (raw) feature set: kept collision columns, generic vehicle summaries, pedestrian fields.

Starting point is :func:`src.data.schema.default_feature_columns` (the leak-free default columns of
the processed collision table). Coded nominal fields become ``category`` (fixed, label-free
vocabulary shared across splits and phases); everything else is float32. The ONS district is
frequency-encoded with an encoder fitted on training rows only and the raw code is dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.schema import default_feature_columns
from src.features.encoders import FrequencyEncoder
from src.features.frame import category_vocab, to_category, to_float32

logger = logging.getLogger(__name__)

DISTRICT_COL = "local_authority_ons_district"
DISTRICT_FREQ_COL = "ons_district_freq"
F0_CATEGORICAL: tuple[str, ...] = (
    "police_force",
    "collision_injury_based",
    "day_of_week",
    "first_road_class",
    "second_road_class",
    "road_type",
    "junction_detail",
    "junction_control",
    "pedestrian_crossing",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "carriageway_hazards",
    "urban_or_rural_area",
    "trunk_road_flag",
    "ped_location",
    "ped_movement",
)


@dataclass(frozen=True)
class F0Columns:
    """Column roles of the F0 set.

    Attributes:
        categorical: Coded nominal columns (``category`` dtype).
        numeric: Numeric columns (float32).
    """

    categorical: tuple[str, ...]
    numeric: tuple[str, ...]


def f0_columns(columns: list[str]) -> F0Columns:
    """Derive the F0 column roles from the processed collision columns.

    Args:
        columns: Processed collision column names.

    Returns:
        :class:`F0Columns` (the raw ONS district is excluded; it is frequency-encoded).

    Raises:
        KeyError: If an expected categorical column is missing.
    """
    defaults = [c for c in default_feature_columns(columns) if c != DISTRICT_COL]
    missing = [c for c in F0_CATEGORICAL if c not in defaults]
    if missing:
        raise KeyError(f"Expected categorical columns absent: {missing}")
    numeric = tuple(c for c in defaults if c not in F0_CATEGORICAL)
    return F0Columns(categorical=F0_CATEGORICAL, numeric=numeric)


def f0_vocab(collisions: pd.DataFrame) -> dict[str, list[object]]:
    """Return the fixed categorical vocabulary of F0 (label-free; all rows may be used).

    Args:
        collisions: Full processed collision frame.

    Returns:
        ``{column: sorted codes}``.
    """
    return category_vocab(collisions, F0_CATEGORICAL)


def district_frequency(rows: pd.DataFrame, train: pd.Series) -> np.ndarray:
    """Frequency-encode the ONS district with an encoder fitted on training rows.

    Args:
        rows: Phase rows with ``local_authority_ons_district``.
        train: Boolean mask of training rows.

    Returns:
        float32 array (share of training rows in the district; unseen -> 0).
    """
    encoder = FrequencyEncoder(name=DISTRICT_COL).fit(rows.loc[train, DISTRICT_COL])
    return encoder.transform(rows[DISTRICT_COL]).to_numpy()


def build_f0(rows: pd.DataFrame, train: pd.Series, vocab: dict[str, list[object]]) -> pd.DataFrame:
    """Build the F0 feature block for one phase.

    Args:
        rows: Phase rows of the processed collision table.
        train: Boolean mask of the rows the frequency encoder may be fitted on.
        vocab: Fixed categorical vocabulary from :func:`f0_vocab`.

    Returns:
        Feature frame aligned with ``rows`` (categoricals first, then numerics, then the
        district frequency).
    """
    roles = f0_columns(list(rows.columns))
    data: dict[str, object] = {}
    for col in roles.categorical:
        data[col] = to_category(rows[col], vocab[col])
    for col in roles.numeric:
        data[col] = to_float32(rows[col])
    data[DISTRICT_FREQ_COL] = district_frequency(rows, train)
    features = pd.DataFrame(data, index=rows.index)
    logger.info(f"F0 block: {features.shape[1]} features ({len(roles.categorical)} categorical, "
                f"{len(roles.numeric) + 1} numeric) on {len(rows):,} rows "
                f"(encoders fitted on {int(train.sum()):,} rows)")
    return features
