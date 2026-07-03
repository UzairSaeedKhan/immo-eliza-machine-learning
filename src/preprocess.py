"""
This file has following functions: 
structural_clean_data(): Deterministic cleaning before train/test split.
build_preprocessor(): Builds a ColumnTransformer pipeline for imputing, encoding, and scaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


COLS_TO_DROP = [
    "property_id",  # unique identifier — no predictive value
    "city",         # 1,395 unique values — redundant with lat/long/province
    "postal_code",  # no linear geographic meaning — lat/long used instead
    "distance_from_train_stations_by_foot", # Reducing noise as they are weak features 
    "distance_from_elementary_school_by_foot", # because they rank below 0.037 correlation
    "distance_from_high_school_by_foot" # with price. 
]

CATEGORICAL_COLS = [
    "province", "type_property", "subtype_property", "state_of_property",
    "heating_type", "sun_exposure", "epc_score", "flooding_area_type",
]

NUMERIC_COLS = [
    "livable_surface", "latitude", "longitude", "facades", "bedrooms",
    "construction_year", "bathrooms", "toilets",
]

BINARY_COLS = ["terrace", "garden", "garage", "swimming_pool"]


def structural_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic cleaning — safe to run on the full dataset before splitting.
    No statistics are learned here, so there is no risk of data leakage.

    Steps:
        - Drop identifier and high-cardinality columns
        - Remove duplicate rows
        - Strip/lowercase all string columns to fix structural text inconsistencies
        - Fix facades == 0 → 1 (no legitimate 0-facade property in raw data)
        - Drop rows with livable_surface < 20 m² (garages / data-entry errors)
    """
    # Merge construction_year before property_id is dropped
    raw_data_path = "./data/raw/scraped_properties.csv"
    df = add_construction_year(df, raw_data_path)
    
    df = df.copy()

    # Drop columns
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])

    # Duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[structural_clean_data] Duplicates removed      : {before - len(df)}")

    # Fix structural text — strip whitespace and lowercase all string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip().str.lower()

    # Fix facades
    df["facades"] = df["facades"].replace(0, 1)

    # Filter implausible livable_surface
    before = len(df)
    df = df[df["livable_surface"] >= 15].reset_index(drop=True)
    print(f"[structural_clean_data] Tiny surface rows removed: {before - len(df)}")

    # # Log-transform price
    # df["price"] = np.log1p(df["price"])

    print(f"[structural_clean_data] Final shape: {df.shape}")
    return df


def build_preprocessor():
    """Builds a ColumnTransformer pipeline for imputing, encoding, and scaling."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer,  NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ], remainder="passthrough")  # passthrough handles BINARY_COLS

def add_construction_year(df: pd.DataFrame, raw_path: str) -> pd.DataFrame:
    """Merges construction_year from raw dataset into cleaned df using property_id."""
    raw = pd.read_csv(raw_path, usecols=["property_id", "construction_year"])
    df = df.merge(raw, on="property_id", how="left")
    print(f"[add_construction_year] Merged. Missing construction_year: {df['construction_year'].isna().sum()}")
    return df