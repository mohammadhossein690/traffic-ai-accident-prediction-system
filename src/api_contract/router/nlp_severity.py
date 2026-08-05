from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

# --- Import Predictor Class ---
# Adjust the import path based on your actual project structure
try:
    from src.ml.nlp.predict import TrafficPredictor
except ImportError:
    TrafficPredictor = None # Set to None if the class cannot be imported
    print("Warning: TrafficPredictor class not found. NLP Severity API might fail.")

# --- API Router Definition ---
router = APIRouter(
    prefix="/nlp", # Base path for this router
    tags=["NLP Severity Prediction"]
)

# --- Request Model Definition ---
# Defines the expected input structure for the NLP Severity API
class TrafficIncidentInput(BaseModel):
    # These fields must exactly match the features your NLP model expects
    Hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    Month: int = Field(..., ge=1, le=12, description="Month of the year (1-12)")
    Weekday: int = Field(..., ge=0, le=6, description="Day of the week (0=Monday, 6=Sunday)")

    Temperature_F: float = Field(..., description="Temperature in Fahrenheit")
    Humidity_pct: float = Field(..., ge=0, le=100, description="Humidity percentage (0-100)")
    Visibility_mi: float = Field(..., ge=0, description="Visibility in miles")
    Wind_Speed_mph: float = Field(..., ge=0, description="Wind speed in miles per hour")

    Description: str = Field(
        ...,
        min_length=1, # Description must not be empty
        description="Detailed description of the traffic incident"
    )

# --- API Endpoint Definition ---
@router.post("/predict")
def predict_severity_endpoint(
    payload: TrafficIncidentInput, # Input data validated by Pydantic
    request: Request               # Access to the request object to get models from app.state
):
    """
    Predicts the severity of a traffic incident based on weather and description.
    """
    # Retrieve the NLP predictor instance from the application state
    # This instance was loaded during application startup in main.py
    predictor = getattr(request.app.state, "nlp_predictor", None)

    # Check if the model was successfully loaded
    if predictor is None:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="NLP Severity model is not initialized or failed to load."
        )

    try:
        # Prepare the input data in the format expected by the model
        # Ensure dictionary keys exactly match model's feature names
        input_data = {
            "Hour": payload.Hour,
            "Month": payload.Month,
            "Weekday": payload.Weekday,
            "Temperature(F)": payload.Temperature_F,
            "Humidity(%)": payload.Humidity_pct,
            "Visibility(mi)": payload.Visibility_mi,
            "Wind_Speed(mph)": payload.Wind_Speed_mph,
        }

        # Call the model's prediction method
        predicted_class = predictor.predict(
            input_data,
            payload.Description # Pass the description separately if required by the model
        )

        # Return the prediction result
        return {
            "status": "success",
            "predicted_severity": int(predicted_class) # Ensure severity is an integer
        }

    except Exception as error:
        # Handle potential errors during the prediction process
        raise HTTPException(
            status_code=500, # Internal Server Error
            detail=f"NLP Severity inference failed: {str(error)}"
        ) from error
