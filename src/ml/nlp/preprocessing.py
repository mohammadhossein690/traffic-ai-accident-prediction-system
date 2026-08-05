from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .text_processor import TextProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "dataset" / "US_Accidents_500k.csv"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "Severity"
TEXT_COLUMN = "Description"

BOOLEAN_COLUMNS = [
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Signal",
]

NUMERIC_COLUMNS = [
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
]

DATETIME_COLUMN = "Start_Time"

STRUCTURED_FEATURE_COLUMNS = [
    "Hour",
    "Month",
    "Weekday",
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "No_Exit",
    "Railway",
    "Roundabout",
    "Station",
    "Stop",
    "Traffic_Signal",
]


def load_dataset(csv_path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_processor: TextProcessor | None = None
) -> pd.DataFrame:
    df = df.copy()

    # Parse datetime safely
    df[DATETIME_COLUMN] = pd.to_datetime(
        df[DATETIME_COLUMN],
        format="mixed",
        errors="coerce"
    )

    # Drop invalid rows for required fields
    df = df.dropna(subset=[DATETIME_COLUMN, TARGET_COLUMN]).copy()

    # Time-based features
    df["Hour"] = df[DATETIME_COLUMN].dt.hour
    df["Month"] = df[DATETIME_COLUMN].dt.month
    df["Weekday"] = df[DATETIME_COLUMN].dt.weekday

    # Fill numeric missing values
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = 0

    # Weather condition fill if column exists
    if "Weather_Condition" in df.columns:
        df["Weather_Condition"] = df["Weather_Condition"].fillna("Unknown")

    # Fill boolean columns and convert to int
    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)
        else:
            df[col] = 0

    # Ensure text column exists
    if TEXT_COLUMN not in df.columns:
        df[TEXT_COLUMN] = ""

    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

    # Clean text
    if text_processor is None:
        text_processor = TextProcessor()

    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(text_processor.clean_text)

    # Keep only rows with valid generated features
    df = df.dropna(subset=["Hour", "Month", "Weekday"]).copy()

    # Ensure final structured columns exist
    for col in STRUCTURED_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    return df


def get_structured_feature_columns() -> List[str]:
    return STRUCTURED_FEATURE_COLUMNS.copy()


def get_text_and_target_columns() -> Tuple[str, str]:
    return TEXT_COLUMN, TARGET_COLUMN
