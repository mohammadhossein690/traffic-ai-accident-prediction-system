from typing import Any, Dict

from .predict import TrafficPredictor


class TrafficPipeline:
    def __init__(self) -> None:
        self.predictor = TrafficPredictor()

    def predict(self, structured_data: Dict[str, Any], description_text: str) -> int:
        return self.predictor.predict(structured_data, description_text)
