from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any

# --- Import Predictor Class ---
# Adjust the import path based on your actual project structure
try:
    from src.duration.predictor import DurationPredictor
except ImportError:
    DurationPredictor = None # Set to None if the class cannot be imported
    print("Warning: DurationPredictor class not found. Duration API might fail.")

# --- API Router Definition ---
router = APIRouter(
    prefix="/api/v1/duration", # Base path for this router
    tags=["Incident Duration Prediction"]
)

# --- Helper function to get the predictor instance ---
def get_duration_predictor(request: Request) -> DurationPredictor:
    """Safely retrieves the duration predictor instance from app state."""
    predictor = getattr(request.app.state, "duration_predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Duration model is not initialized or failed to load."
        )
    return predictor

# --- Request Model Definition ---
# Defines the expected input structure for the Duration API
class DurationInput(BaseModel):
    # These fields must match the inputs expected by the DurationPredictor.predict method
    # and align with the features used during training.

    Start_Time: str = Field(..., description="Start time of the incident (e.g., 'YYYY-MM-DD HH:MM:SS')")
    Severity: int   # Severity predicted by another model or provided externally

    # Features directly used by the model pipeline
    # Use Field(alias=...) if the incoming JSON keys differ from the internal model feature names
    Distance_mi: float = Field(alias="Distance(mi)")
    Temperature_F: float = Field(alias="Temperature(F)")
    Humidity_pct: float = Field(alias="Humidity(%)")
    Visibility_mi: float = Field(alias="Visibility(mi)")
    Wind_Speed_mph: float = Field(alias="Wind_Speed(mph)")
    Pressure_in: float = Field(alias="Pressure(in)")
    Precipitation_in: float = Field(alias="Precipitation(in)")

    # Binary features (often One-Hot Encoded or treated as categories)
    Amenity: bool
    Bump: bool
    Crossing: bool
    Give_Way: bool
    Junction: bool
    No_Exit: bool
    Railway: bool
    Roundabout: bool
    Station: bool
    Stop: bool
    Traffic_Calming: bool
    Traffic_Signal: bool

    # Categorical features
    Weather_Condition: str
    State: str
    Sunrise_Sunset: str # e.g., "Day", "Night"


@router.post("/predict")
def predict_duration_endpoint(
    payload: DurationInput, # Input data validated by Pydantic
    request: Request        # Access to the request object to get models from app.state
):
    """
    Predicts the duration of a traffic incident.
    """
    predictor = get_duration_predictor(request)

    try:
        # Convert Pydantic model to a dictionary.
        # model_dump(by_alias=True) ensures that aliases (like "Distance(mi)") are used as keys.
        data = payload.model_dump(by_alias=True)

        # IMPORTANT NOTE:
        # The DurationPredictor.predict method internally extracts time-based features
        # (Hour, Month, Weekday, Year, Day, Is_Weekend) from the 'Start_Time' field.
        # Therefore, you only need to pass the raw 'Start_Time' and other required features.
        # Do NOT manually add Hour, Month, etc., to the `data` dictionary here if the predictor handles it.
        # Verify this behavior against your src/duration/predictor.py file.

        # Call the model's prediction method
        prediction = predictor.predict(data)

        # Process the prediction result into a more readable format
        prediction_minutes = round(float(prediction), 2)
        hours = int(prediction_minutes // 60)
        remaining_minutes = round(prediction_minutes % 60)

        return {
            "status": "success",
            "predicted_duration_minutes": prediction_minutes,
            "predicted_duration_readable": f"{hours}h {remaining_minutes}m",
        }

    except Exception as e:
        # Handle potential errors during prediction
        raise HTTPException(
            status_code=400, # Bad Request if input is problematic or prediction fails
            detail=f"Duration prediction error: {str(e)}"
        ) from e
