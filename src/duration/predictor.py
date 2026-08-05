import os
import pickle
import pandas as pd
import numpy as np

class DurationPredictor:
    def __init__(self, model_path: str = r"C:\Users\NoteBook\Desktop\traffic_ai_system\models\duration\duration_pipeline.pkl"):
        self.model_path = model_path
        self.pipeline = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}. Please run trainer.py first.")
        
        with open(self.model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        print("Duration prediction pipeline loaded successfully.")

    def predict(self, input_data: dict) -> float:
        """
        Accepts a dictionary of input features, processes them, and returns predicted duration in minutes.
        """
        # Convert dictionary input to DataFrame (1 row)
        df = pd.DataFrame([input_data])

        # Convert Start_Time to datetime and extract features
        start_time = pd.to_datetime(df["Start_Time"]).iloc[0]
        
        df["Hour"] = start_time.hour
        df["Month"] = start_time.month
        df["Weekday"] = start_time.weekday()
        df["Year"] = start_time.year
        df["Day"] = start_time.day
        df["Is_Weekend"] = 1 if start_time.weekday() >= 5 else 0

        # Predict using the loaded pipeline
        prediction = self.pipeline.predict(df)
        
        # Ensure predicted duration is non-negative
        predicted_minutes = float(np.clip(prediction[0], a_min=3.0, a_max=776.0))
        return predicted_minutes
