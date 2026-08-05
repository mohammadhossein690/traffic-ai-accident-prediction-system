import uvicorn # type: ignore
import os

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Import Routers for each API
from .router.vision_prediction import router as vision_router
from .router.nlp_severity import router as nlp_severity_router
from .router.incident_duration import router as incident_duration_router

# Import Model Predictor classes
# Adjust import paths based on your actual project structure
try:
    # Assuming CNN model prediction logic is encapsulated in a class
    # If model is loaded directly in vision_prediction.py, this might not be needed here.
    # from src.ml.vision.predict import TrafficIncidentClassifier
    VISION_MODEL_LOADED = True # Placeholder, adjust if using a separate predict class
    print("Vision model loading logic noted.")
except ImportError:
    VISION_MODEL_LOADED = False
    print("Warning: Vision model predictor class not found. CNN API might not function correctly.")

try:
    # NLP Severity predictor class
    from src.ml.nlp.predict import TrafficPredictor as NLPPredictor # type: ignore
except ImportError:
    NLPPredictor = None
    print("Warning: NLP Severity model predictor class not found. NLP Severity API might not function correctly.")

try:
    # Duration predictor class
    from src.duration.predictor import DurationPredictor # type: ignore
except ImportError:
    DurationPredictor = None
    print("Warning: Duration model predictor class not found. Duration API might not function correctly.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Traffic AI Prediction API",
    description=(
        "API for detecting traffic incidents (CNN), predicting severity (NLP), "
        "and estimating duration (XGBoost)."
    ),
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading Event Handler ---
@app.on_event("startup")
def load_ml_models():
    """Loads machine learning models into application state on startup."""
    print("Attempting to load ML models...")

    # Load Vision Model (if using a separate predictor class)
    # if VISION_MODEL_LOADED:
    #     try:
    #         app.state.vision_classifier = TrafficIncidentClassifier()
    #         print("Vision model loaded successfully.")
    #     except Exception as e:
    #         print(f"Error loading Vision model: {e}")
    #         # Decide if this error should stop the application startup
    #         # raise e

    # Load NLP Severity Model
    if NLPPredictor:
        try:
            app.state.nlp_predictor = NLPPredictor()
            print("NLP Severity Predictor loaded successfully.")
        except Exception as e:
            print(f"Error loading NLP Severity model: {e}")
            raise e # Stop startup if critical model fails

    # Load Duration Model
    if DurationPredictor:
        try:
            # Assumes DurationPredictor constructor handles loading the pipeline internally
            app.state.duration_predictor = DurationPredictor()
            print("Duration Predictor loaded successfully.")
        except Exception as e:
            print(f"Error loading Duration model: {e}")
            raise e # Stop startup if critical model fails


# --- Include Routers ---
# These routers define the API endpoints and their logic
app.include_router(vision_router)
app.include_router(nlp_severity_router)
app.include_router(incident_duration_router)


# --- Root Endpoint ---
@app.get("/")
def root():
    """Root endpoint to confirm API is running."""
    return {
        "message": "Traffic Incident Analysis API is running"
    }
@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


# --- Running the API (for local development) ---
if __name__ == "__main__":
    # To run this application locally:
    # 1. Navigate to the root directory of your project (e.g., traffic_ai_system/)
    # 2. Run the command: uvicorn src.api_contract.main:app --reload
    print("To run locally, use: uvicorn src.api_contract.main:app --reload")
    # Note: uvicorn.run(...) is typically used when the file is the main entry point,
    # but here we instruct the user on how to run it via the uvicorn command.
