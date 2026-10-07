"""Load the Kaggle Adidas US sales dataset and fix its revenue figures."""
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_DATASET = "heemalichaudhari/adidas-sales-dataset"
FILE_NAME = "Adidas US Sales Datasets.xlsx"

COLUMNS = {
    "Retailer": "retailer", "Retailer ID": "retailer_id", "Invoice Date": "date",
    "Region": "region", "State": "state", "City": "city", "Product": "product",
    "Price per Unit": "price", "Units Sold": "units", "Total Sales": "reported_sales",
    "Operating Profit": "reported_profit", "Operating Margin": "operating_margin",
    "Sales Method": "sales_method",
}


def dataset_path():
    """Download the dataset with kagglehub (cached after the first call)."""
    import kagglehub

    return Path(kagglehub.dataset_download(KAGGLE_DATASET)) / FILE_NAME


def load_raw(path=None):
    """The spreadsheet as published (the table starts on row 5)."""
    raw = pd.read_excel(path or dataset_path(), header=4)
    return raw.drop(columns=[c for c in raw.columns if str(c).startswith("Unnamed")])


def audit_reported_sales(raw):
    """How reported sales compare with price x units."""
    ratio = raw["Total Sales"] / (raw["Price per Unit"] * raw["Units Sold"])
    label = np.select([np.isclose(ratio, 1), np.isclose(ratio, 10)],
                      ["price x units", "10 x price x units"], "other")
    return pd.Series(label, index=raw.index, name="reported sales equal")


def clean(raw):
    """Consistent revenue and profit, one row per invoice line, sorted by date.

    In 40% of the rows the reported sales are exactly ten times price x units
    (all in-store sales and part of the rest), and the reported profit inherits
    the error because it equals sales x margin. Revenue is rebuilt as
    price x units and profit as revenue x margin. The margin is a ratio and is
    not affected.
    """
    df = raw.rename(columns=COLUMNS)
    df["sales_inflated"] = audit_reported_sales(raw).eq("10 x price x units")
    df["revenue"] = df["price"] * df["units"]
    df["profit"] = df["revenue"] * df["operating_margin"]
    df["month"] = df["date"].dt.month
    return df.sort_values("date", kind="stable").reset_index(drop=True)
