"""K-means segmentation of city-product markets."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["avg_price", "avg_units", "avg_margin", "online_share", "log_revenue"]


def market_profiles(df):
    """One row per city-product market: price level, deal size, margin,
    channel mix and size."""
    out = (df.assign(online=df["sales_method"].eq("Online"))
             .groupby(["city", "product"])
             .agg(avg_price=("price", "mean"), avg_units=("units", "mean"),
                  avg_margin=("operating_margin", "mean"),
                  online_share=("online", "mean"), revenue=("revenue", "sum")))
    out["log_revenue"] = np.log(out["revenue"])
    return out


def silhouette_by_k(profiles, ks=range(2, 9)):
    """Silhouette score of K-means for each number of clusters."""
    x = StandardScaler().fit_transform(profiles[FEATURES])
    return pd.Series({k: silhouette_score(
        x, KMeans(k, n_init=10, random_state=SEED).fit_predict(x)) for k in ks},
        name="silhouette").rename_axis("k")


def fit_segments(profiles, k):
    """Scaler + K-means fitted on the given markets."""
    return make_pipeline(StandardScaler(), KMeans(k, n_init=10, random_state=SEED)).fit(
        profiles[FEATURES])


def assign(df, profiles, segmenter):
    """Segment of each sale's market; -1 for markets unseen when fitting."""
    labels = pd.Series(segmenter.predict(profiles[FEATURES]), index=profiles.index)
    keys = pd.MultiIndex.from_frame(df[["city", "product"]])
    return pd.Series(labels.reindex(keys).fillna(-1).astype(int).to_numpy(),
                     index=df.index, name="segment")
