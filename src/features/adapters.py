"""Per-model-family preprocessing adapters.

Every feature builder returns a pandas DataFrame whose categorical columns have ``category`` dtype
and whose numeric columns are floats (NaN = missing). Each model family needs a different view:

* ``onehot``  (Logistic Regression): one-hot categoricals (rare levels pooled), median imputation
  with missing indicators, standard scaling.
* ``ordinal`` (Decision Tree, Random Forest): integer codes for categoricals (missing = -1),
  numerics left with NaN (scikit-learn trees support missing values natively).
* ``native``  (XGBoost, LightGBM, CatBoost): the DataFrame itself with ``category`` dtype.

Adapters are fitted on training rows only and then applied unchanged to TUNE / CAL / TEST.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

MODEL_ADAPTER: dict[str, str] = {
    "lr": "onehot",
    "dt": "ordinal",
    "rf": "ordinal",
    "xgb": "native",
    "lgbm": "native",
    "cat": "native",
}

MISSING_LEVEL = "__missing__"


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split columns into categorical and numeric lists based on dtype.

    Args:
        df: Feature frame.

    Returns:
        Tuple ``(categorical_columns, numeric_columns)``.
    """
    cat_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    num_cols = [c for c in df.columns if c not in cat_cols]
    return cat_cols, num_cols


@dataclass
class FeatureAdapter:
    """Fit-on-train, transform-anywhere preprocessing for one model family.

    Attributes:
        kind: One of ``onehot``, ``ordinal`` or ``native``.
        min_frequency: Minimum count for a one-hot level (rarer levels are pooled).
        num_fill: Optional numeric imputation values applied to every split before the
            family-specific transform (set when training data were SMOTE-NC resampled).
    """

    kind: str
    min_frequency: int = 50
    num_fill: dict[str, float] | None = None
    cat_cols: list[str] = field(default_factory=list)
    num_cols: list[str] = field(default_factory=list)
    categories: dict[str, list] = field(default_factory=dict)
    _pipeline: ColumnTransformer | None = None

    def fit(self, X: pd.DataFrame) -> "FeatureAdapter":
        """Learn column roles, category vocabularies and (for ``onehot``) the sklearn pipeline.

        Args:
            X: Training feature frame.

        Returns:
            The fitted adapter.
        """
        self.cat_cols, self.num_cols = split_columns(X)
        self.categories = {
            c: [v for v in X[c].cat.categories if v != MISSING_LEVEL] for c in self.cat_cols
        }
        if self.kind == "onehot":
            self._pipeline = self._build_onehot_pipeline()
            self._pipeline.fit(self._prepare_onehot_input(X))
        logger.debug(
            "Adapter %s fitted: %d categorical, %d numeric", self.kind, len(self.cat_cols),
            len(self.num_cols),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        """Transform a feature frame into the model-family representation.

        Args:
            X: Feature frame with the same columns as the training frame.

        Returns:
            Model-ready matrix (DataFrame for ``native``/``ordinal``, array for ``onehot``).
        """
        X = X[self.cat_cols + self.num_cols]
        if self.num_fill:
            X = X.fillna({c: v for c, v in self.num_fill.items() if c in self.num_cols})
        if self.kind == "native":
            return self._align_categories(X)
        if self.kind == "ordinal":
            return self._to_ordinal(X)
        if self.kind == "onehot":
            assert self._pipeline is not None, "Adapter not fitted"
            return self._pipeline.transform(self._prepare_onehot_input(X)).astype(np.float32)
        raise ValueError(f"Unknown adapter kind: {self.kind}")

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        """Fit on ``X`` and transform it.

        Args:
            X: Training feature frame.

        Returns:
            Model-ready training matrix.
        """
        return self.fit(X).transform(X)

    def _align_categories(self, X: pd.DataFrame) -> pd.DataFrame:
        """Force categorical columns onto the training vocabulary (unseen levels -> NaN)."""
        X = X.copy()
        for c in self.cat_cols:
            values = X[c].astype(object)
            values = values.where(values.isin(self.categories[c]), None)  # unseen/missing -> NaN
            X[c] = pd.Categorical(values, categories=self.categories[c])
        for c in self.num_cols:
            X[c] = X[c].astype(np.float32)
        return X

    def _to_ordinal(self, X: pd.DataFrame) -> pd.DataFrame:
        """Integer-code categoricals with the training vocabulary; missing/unseen -> -1."""
        X = self._align_categories(X)
        for c in self.cat_cols:
            X[c] = X[c].cat.codes.astype(np.float32)
        return X

    def _prepare_onehot_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Represent categoricals as strings with an explicit missing level for one-hot."""
        X = self._align_categories(X)
        for c in self.cat_cols:
            X[c] = X[c].astype(object).where(X[c].notna(), MISSING_LEVEL).astype(str)
        return X

    def _build_onehot_pipeline(self) -> ColumnTransformer:
        """Create the one-hot / impute / scale column transformer."""
        cat_pipe = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=self.min_frequency,
            sparse_output=False,
            dtype=np.float32,
        )
        num_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                         keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]
        )
        return ColumnTransformer(
            [("cat", cat_pipe, self.cat_cols), ("num", num_pipe, self.num_cols)],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def feature_names(self) -> list[str]:
        """Return output feature names of the transformed matrix.

        Returns:
            List of output column names.
        """
        if self.kind == "onehot" and self._pipeline is not None:
            return list(self._pipeline.get_feature_names_out())
        return self.cat_cols + self.num_cols


def make_adapter(model_name: str) -> FeatureAdapter:
    """Create the adapter appropriate for a model.

    Args:
        model_name: Registry key (``lr``, ``dt``, ``rf``, ``xgb``, ``lgbm``, ``cat``).

    Returns:
        Unfitted adapter.
    """
    return FeatureAdapter(kind=MODEL_ADAPTER[model_name])
