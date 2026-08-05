from typing import List, Tuple

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
from sklearn.feature_extraction.text import CountVectorizer # type: ignore


def prepare_structured_features(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    X_structured = df[feature_cols].copy()

    for col in feature_cols:
        if col not in X_structured.columns:
            X_structured[col] = 0

    X_structured = X_structured[feature_cols].fillna(0)
    return X_structured


def fit_scaler_and_transform(
    X_structured: pd.DataFrame
) -> Tuple[StandardScaler, csr_matrix]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_structured)
    return scaler, csr_matrix(X_scaled)


def transform_structured_features(
    X_structured: pd.DataFrame,
    scaler: StandardScaler
) -> csr_matrix:
    X_scaled = scaler.transform(X_structured)
    return csr_matrix(X_scaled)


def fit_vectorizer_and_transform(
    text_series: pd.Series,
    max_features: int = 5000,
    ngram_range: tuple = (1, 2)
) -> Tuple[CountVectorizer, csr_matrix]:
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )
    X_text = vectorizer.fit_transform(text_series)
    return vectorizer, X_text


def transform_text_features(
    text_series: pd.Series,
    vectorizer: CountVectorizer
) -> csr_matrix:
    return vectorizer.transform(text_series)


def combine_features(
    X_structured,
    X_text
):
    return hstack([X_structured, X_text])


def fit_label_encoder(target_series: pd.Series) -> Tuple[LabelEncoder, pd.Series]:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(target_series)
    return label_encoder, y_encoded


def transform_target(
    target_series: pd.Series,
    label_encoder: LabelEncoder
):
    return label_encoder.transform(target_series)
