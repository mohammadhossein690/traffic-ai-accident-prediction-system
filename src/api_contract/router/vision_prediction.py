from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from pathlib import Path
import shutil
import os
import numpy as np
import tensorflow as tf

# --- Configuration ---
# Get the directory of the current file
CURRENT_DIR = Path(__file__).resolve()
# Determine the base directory of the project (assuming structure: src/api_contract/router/...)
# Go up 4 levels: file -> router -> api_contract -> src -> project root
BASE_DIR = CURRENT_DIR.parent.parent.parent.parent

# Define the path to the model file
MODEL_DIR = BASE_DIR / "models" / "vision"
MODEL_PATH = MODEL_DIR / "traffic_incident_model.keras"

# Define directory for temporary uploads
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True) # Create uploads directory if it doesn't exist


# --- Model Loading ---
# Load the model directly here or ensure it's loaded via app.state in main.py
# For simplicity, loading directly here. If you prefer app.state, move this to main.py
# and access via request.app.state.vision_model
model = None
if not MODEL_PATH.exists():
    print(f"Warning: Vision model file not found at {MODEL_PATH}. Vision API may fail.")
else:
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress TF verbose logs
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        tf.get_logger().setLevel("ERROR")

        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            custom_objects={
                "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
            },
            compile=False # Compile=False if you don't need to fine-tune or evaluate loss here
        )
        print(f"Vision model loaded successfully from {MODEL_PATH}.")
    except Exception as e:
        print(f"Error loading Vision model: {e}")
        model = None # Ensure model is None if loading fails


# --- API Router Definition ---
router = APIRouter(
    prefix="/api/v1/vision",
    tags=["Vision Prediction (CNN)"]
)

# --- Prediction Logic ---
def predict_image(image_path: str) -> dict:
    """Performs image prediction using the loaded TensorFlow model."""
    if model is None:
        raise RuntimeError("Vision model is not loaded or failed to load.")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load image, resize to target_size (224x224 for MobileNetV2)
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    # Convert image to NumPy array and add batch dimension
    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Run prediction. Model outputs probability of the 'Nat' class.
    # The index [0][0] assumes the output is a single probability value.
    prob_nat = float(model.predict(image_array, verbose=0)[0][0])

    # Accident probability is the complement
    prob_acc = 1.0 - prob_nat

    # Determine the predicted class based on a 0.5 threshold
    if prob_nat >= 0.5:
        predicted_class = "Nat"
        confidence = prob_nat
    else:
        predicted_class = "Acc"
        confidence = prob_acc

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "accident_detected": predicted_class == "Acc",
        "accident_probability": round(prob_acc, 4),
        "non_accident_probability": round(prob_nat, 4)
    }

# --- API Endpoint Definition ---
@router.post("/predict")
async def predict_accident_endpoint(
    request: Request, # Parameters without default values must come first
    file: UploadFile = File(...) # Parameters with default values must come after
):
# Process the vision prediction logic here
    ...

    """
    Endpoint to upload an image for traffic incident detection.
    Accepts PNG, JPG, JPEG formats.
    """
    # Validate file extension
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPG, and JPEG image formats are supported."
        )

    # Define temporary path to save the uploaded file
    file_path = UPLOAD_DIR / file.filename

    try:
        # Save the uploaded file temporarily
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Perform prediction on the saved image
        result = predict_image(str(file_path))

        return {
            "filename": file.filename,
            **result # Unpack the prediction results dictionary
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
         raise HTTPException(status_code=503, detail=str(e)) # Service Unavailable if model not loaded
    except Exception as e:
        # Catch any other unexpected errors during processing
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during inference: {str(e)}"
        )

    finally:
        # Clean up the temporary file after processing
        if file_path.exists():
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error removing temporary file {file_path}: {e}")
