from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.utils.class_weight import compute_sample_weight # type: ignore
from xgboost import XGBClassifier

from .feature_extraction import (
    combine_features,
    fit_label_encoder,
    fit_scaler_and_transform,
    fit_vectorizer_and_transform,
    prepare_structured_features,
)
from .preprocessing import (
    MODELS_DIR,
    TARGET_COLUMN,
    TEXT_COLUMN,
    get_structured_feature_columns,
    load_dataset,
    preprocess_dataframe,
)
from .text_processor import TextProcessor


def train_and_save() -> None:
    print("Loading dataset...")
    df = load_dataset()
    print(f"Original dataset shape: {df.shape}")

    print("Preprocessing data...")
    text_processor = TextProcessor()
    df = preprocess_dataframe(df, text_processor=text_processor)
    print(f"Preprocessed dataset shape: {df.shape}")

    feature_cols = get_structured_feature_columns()
    print(f"Structured features: {feature_cols}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Preparing structured features...")
    X_train_structured_df = prepare_structured_features(X_train, feature_cols)
    scaler, X_train_structured = fit_scaler_and_transform(X_train_structured_df)

    print("Preparing text features...")
    vectorizer, X_train_text = fit_vectorizer_and_transform(X_train[TEXT_COLUMN])

    print("Combining features...")
    X_train_combined = combine_features(X_train_structured, X_train_text)

    print("Encoding target...")
    label_encoder, y_train_encoded = fit_label_encoder(y_train)

    print("Calculating balanced sample weights...")
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train_encoded
    )

    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        objective="multi:softprob",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(X_train_combined, y_train_encoded, sample_weight=sample_weights)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Saving artifacts...")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")
    joblib.dump(model, MODELS_DIR / "xgb_model.joblib")

    print("Training completed successfully.")
    print(f"Artifacts saved in: {MODELS_DIR}")


if __name__ == "__main__":
    train_and_save()
