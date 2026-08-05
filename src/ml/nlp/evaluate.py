import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

from .feature_extraction import (
    combine_features,
    prepare_structured_features,
    transform_structured_features,
    transform_text_features,
)
from .preprocessing import (
    MODELS_DIR,
    TARGET_COLUMN,
    TEXT_COLUMN,
    load_dataset,
    preprocess_dataframe,
)
from .text_processor import TextProcessor


def evaluate_model() -> None:
    print("Loading artifacts...")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    vectorizer = joblib.load(MODELS_DIR / "vectorizer.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    model = joblib.load(MODELS_DIR / "xgb_model.joblib")

    print("Loading dataset...")
    df = load_dataset()

    print("Preprocessing data...")
    text_processor = TextProcessor()
    df = preprocess_dataframe(df, text_processor=text_processor)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Same split logic as train.py
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Preparing evaluation features...")
    X_test_structured_df = prepare_structured_features(X_test_eval, feature_cols)
    X_test_structured = transform_structured_features(X_test_structured_df, scaler)
    X_test_text = transform_text_features(X_test_eval[TEXT_COLUMN], vectorizer)
    X_test_combined = combine_features(X_test_structured, X_test_text)

    print("Running predictions...")
    y_pred_encoded = model.predict(X_test_combined)

    y_test_encoded = label_encoder.transform(y_test_eval)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    print("\nClassification Report:\n")
    print(classification_report(y_test_eval, y_pred))

    cm = confusion_matrix(y_test_eval, y_pred, labels=label_encoder.classes_)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    output_path = MODELS_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nConfusion matrix image saved to: {output_path}")


if __name__ == "__main__":
    evaluate_model()
