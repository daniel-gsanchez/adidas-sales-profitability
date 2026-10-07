"""Operating-margin models, evaluated out of time."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TEST_START = pd.Timestamp("2021-10-01")  # last quarter is held out
CATEGORICAL = ["sales_method", "product", "retailer", "region", "state", "city"]
NUMERIC = ["price", "units", "month"]


def time_split(df):
    test = df["date"] >= TEST_START
    return ~test, test


def preprocessor(categorical, numeric):
    """One-hot categories (unseen ones are ignored) and standardized numbers."""
    return ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numbers", StandardScaler(), numeric),
    ])


def ridge(categorical=CATEGORICAL, numeric=NUMERIC):
    """Additive linear baseline: no interactions between attributes."""
    return make_pipeline(preprocessor(categorical, numeric), Ridge(alpha=1.0))


def random_forest(categorical=CATEGORICAL, numeric=NUMERIC, **params):
    params = {"n_estimators": 300, "n_jobs": -1, "random_state": SEED, **params}
    return make_pipeline(preprocessor(categorical, numeric),
                         RandomForestRegressor(**params))


def tune_random_forest(x, y, n_iter=20):
    """Randomized search with time-ordered folds inside the training period."""
    space = {
        "randomforestregressor__max_depth": [None, 10, 20, 30],
        "randomforestregressor__min_samples_leaf": [1, 2, 5, 10],
        "randomforestregressor__max_features": ["sqrt", 0.3, 0.6, 1.0],
    }
    search = RandomizedSearchCV(random_forest(), space, n_iter=n_iter,
                                cv=TimeSeriesSplit(4), scoring="r2",
                                random_state=SEED, n_jobs=1)
    return search.fit(x, y)


def scores(y, pred):
    """R2 and errors in margin percentage points (pp)."""
    return {"R2": r2_score(y, pred),
            "MAE (pp)": 100 * mean_absolute_error(y, pred),
            "RMSE (pp)": 100 * np.sqrt(mean_squared_error(y, pred))}
