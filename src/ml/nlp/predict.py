from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from scipy.sparse import hstack

from .preprocessing import MODELS_DIR, TEXT_COLUMN
from .text_processor import TextProcessor


class TrafficPredictor:
    def __init__(self, models_dir: Path = MODELS_DIR / "nlp") -> None:
        self.models_dir = models_dir

        self.scaler = joblib.load(self.models_dir / "scaler.joblib")
        self.vectorizer = joblib.load(self.models_dir / "vectorizer.joblib")
        self.label_encoder = joblib.load(
            self.models_dir / "label_encoder.joblib"
        )
        self.feature_cols = joblib.load(
            self.models_dir / "feature_cols.joblib"
        )
        self.model = joblib.load(self.models_dir / "xgb_model.joblib")

        self.text_processor = TextProcessor()

    def _prepare_structured_input(
        self,
        structured_data: Dict[str, Any]
    ) -> pd.DataFrame:
        row = {}

        for col in self.feature_cols:
            row[col] = structured_data.get(col, 0)

        df = pd.DataFrame([row])
        df = df[self.feature_cols].fillna(0)

        return df

    def _prepare_text_input(self, text: str):
        cleaned_text = self.text_processor.clean_text(text)

        return self.vectorizer.transform([cleaned_text])

    def predict(
        self,
        structured_data: Dict[str, Any],
        description_text: str
    ) -> int:
        structured_df = self._prepare_structured_input(structured_data)
        structured_scaled = self.scaler.transform(structured_df)

        text_vector = self._prepare_text_input(description_text)

        X_input = hstack([structured_scaled, text_vector])

        pred_encoded = self.model.predict(X_input)[0]
        pred_label = self.label_encoder.inverse_transform([pred_encoded])[0]

        return int(pred_label)
