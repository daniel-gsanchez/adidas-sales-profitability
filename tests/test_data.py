import pandas as pd

from adidas.data import clean
from adidas.segmentation import assign, fit_segments, market_profiles


def toy_raw():
    """Two consistent rows and one with reported sales inflated 10x."""
    return pd.DataFrame({
        "Retailer": ["A", "A", "B"], "Retailer ID": [1, 1, 2],
        "Invoice Date": pd.to_datetime(["2021-01-02", "2021-01-01", "2021-01-03"]),
        "Region": ["West"] * 3, "State": ["CA"] * 3, "City": ["LA", "LA", "SF"],
        "Product": ["Shoes"] * 3, "Price per Unit": [50.0, 40.0, 60.0],
        "Units Sold": [10, 5, 2], "Total Sales": [500.0, 200.0, 1200.0],
        "Operating Profit": [250.0, 60.0, 480.0], "Operating Margin": [0.5, 0.3, 0.4],
        "Sales Method": ["Online", "Outlet", "In-store"],
    })


def test_clean_rebuilds_revenue_and_profit():
    df = clean(toy_raw())
    assert df["date"].is_monotonic_increasing
    assert df["revenue"].tolist() == [200.0, 500.0, 120.0]
    assert df["profit"].tolist() == [60.0, 250.0, 48.0]  # revenue x margin
    assert df["sales_inflated"].tolist() == [False, False, True]


def test_unseen_markets_get_their_own_segment():
    df = clean(toy_raw())
    profiles = market_profiles(df[df["city"] == "LA"])
    segmenter = fit_segments(profiles, k=1)
    assert assign(df, profiles, segmenter).tolist() == [0, 0, -1]
