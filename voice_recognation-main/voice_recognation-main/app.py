from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pickle
import numpy as np
from audio_utils import extract_features_from_filelike

MODEL_PATH = "bangla_speech_models.pkl"

app = FastAPI(
    title="Bangla Speech Gender & Region API",
    description="API for predicting gender and region from Bangla speech audio files",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load models once at startup
with open(MODEL_PATH, "rb") as f:
    ensemble_obj = pickle.load(f)

# Extract models and encoders based on actual pickle structure
if not isinstance(ensemble_obj, dict):
    raise TypeError(f"Expected pickle file to contain a dictionary, but got {type(ensemble_obj)}")

# Get models and encoders - handle both old and new key names
gender_model = ensemble_obj.get("gender_model") or ensemble_obj.get("models")
region_model = ensemble_obj.get("region_model")
gender_encoder = ensemble_obj.get("le_gender") or ensemble_obj.get("gender_encoder")
region_encoder = ensemble_obj.get("le_region") or ensemble_obj.get("region_encoder")

# Check if models are loaded (warn but don't fail at startup)
available_keys = list(ensemble_obj.keys())
models_loaded = all([gender_model is not None, region_model is not None, 
                     gender_encoder is not None, region_encoder is not None])

if not models_loaded:
    missing = []
    if gender_model is None:
        missing.append("gender_model")
    if region_model is None:
        missing.append("region_model")
    if gender_encoder is None:
        missing.append("le_gender")
    if region_encoder is None:
        missing.append("le_region")
    print(f"WARNING: Models not properly loaded. Missing or None: {', '.join(missing)}")
    print(f"Available keys in pickle file: {available_keys}")
    print(f"Please ensure the models and encoders are properly saved to {MODEL_PATH}")

def predict(features: np.ndarray):
    """
    Predict gender and region using separate models.
    """
    # Check if models are loaded before prediction
    if gender_model is None or region_model is None or gender_encoder is None or region_encoder is None:
        missing = []
        if gender_model is None:
            missing.append("gender_model")
        if region_model is None:
            missing.append("region_model")
        if gender_encoder is None:
            missing.append("le_gender")
        if region_encoder is None:
            missing.append("le_region")
        raise ValueError(
            f"Models not loaded. The following components are None: {', '.join(missing)}. "
            f"Please ensure the models and encoders are properly saved to {MODEL_PATH}"
        )
    
    # Predict gender
    gender_pred = gender_model.predict(features)
    # Predict region
    region_pred = region_model.predict(features)
    
    # Decode labels if using label encoders
    # Handle both single predictions and array predictions
    if hasattr(gender_pred, '__len__') and len(gender_pred) > 0:
        gender_class = gender_pred[0] if isinstance(gender_pred, np.ndarray) else gender_pred
    else:
        gender_class = gender_pred
    
    if hasattr(region_pred, '__len__') and len(region_pred) > 0:
        region_class = region_pred[0] if isinstance(region_pred, np.ndarray) else region_pred
    else:
        region_class = region_pred
    
    # Decode labels using label encoders
    gender_label = gender_encoder.inverse_transform([gender_class])[0]
    region_label = region_encoder.inverse_transform([region_class])[0]

    return gender_label, region_label

@app.get("/")
async def root():
    """Root endpoint with API information."""
    models_status = "loaded" if models_loaded else "not loaded"
    return {
        "message": "Bangla Speech Gender & Region API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information (this endpoint)",
            "/docs": "Interactive API documentation (Swagger UI)",
            "/redoc": "Alternative API documentation (ReDoc)",
            "/health": "Health check and model status",
            "/predict": "POST endpoint for audio prediction"
        },
        "models_status": models_status,
        "usage": "Use POST /predict with an audio file to get gender and region predictions"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint to verify models are loaded."""
    models_status = {
        "gender_model": gender_model is not None,
        "region_model": region_model is not None,
        "gender_encoder": gender_encoder is not None,
        "region_encoder": region_encoder is not None
    }
    all_loaded = all(models_status.values())
    
    return {
        "status": "healthy" if all_loaded else "degraded",
        "models_loaded": all_loaded,
        "components": models_status,
        "ready_for_predictions": all_loaded
    }

@app.post("/predict")
async def predict_speaker(audio: UploadFile = File(...)):
    # Basic validation
    if not audio.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    # Read file into memory
    raw_bytes = await audio.read()

    # Extract features (must match your training pipeline)
    try:
        import io
        features = extract_features_from_filelike(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

    # Run prediction
    try:
        gender, region = predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "gender": gender,
        "region": region,
        "file_name": audio.filename,
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
