from pathlib import Path
import os

# Suppress TensorFlow verbose logs BEFORE importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from PIL import Image


# Optional: reduce TensorFlow Python-side warnings/logging
tf.get_logger().setLevel("ERROR")


class AccidentImagePredictor:
    """
    Predicts whether an image contains an accident.

    Class mapping from the training dataset:
        0 -> Acc
        1 -> Nat

    The saved model has a sigmoid output:
        output = P(Nat)

    Therefore:
        P(Acc) = 1 - P(Nat)
    """

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)

        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={
                "preprocess_input": (
                    tf.keras.applications.mobilenet_v2.preprocess_input
                )
            },
            compile=False,
        )

        self.class_names = ["Acc", "Nat"]
        self.img_size = (224, 224)

    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize(self.img_size)
        image_array = np.array(img, dtype=np.float32)
        return np.expand_dims(image_array, axis=0)

    def predict(self, image: Image.Image) -> dict:
        x = self._prepare_image(image)

        prob_nat = float(self.model.predict(x, verbose=0)[0][0])
        prob_acc = 1.0 - prob_nat

        predicted_class = "Nat" if prob_nat >= 0.5 else "Acc"
        confidence = prob_nat if predicted_class == "Nat" else prob_acc

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "accident_detected": predicted_class == "Acc",
            "accident_probability": round(prob_acc, 4),
            "non_accident_probability": round(prob_nat, 4),
        }
