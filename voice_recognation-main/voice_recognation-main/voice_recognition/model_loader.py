"""
Load ML models once at Django startup.
"""
import pickle
import os
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, "bangla_speech_models.pkl")

# Global variables to store loaded models
gender_model = None
region_model = None
gender_encoder = None
region_encoder = None
models_loaded = False

def load_models():
    """Load models from pickle file."""
    global gender_model, region_model, gender_encoder, region_encoder, models_loaded
    
    try:
        with open(MODEL_PATH, "rb") as f:
            ensemble_obj = pickle.load(f)
        
        if not isinstance(ensemble_obj, dict):
            print(f"ERROR: Expected pickle file to contain a dictionary, but got {type(ensemble_obj)}")
            return
        
        # Get models and encoders
        gender_model = ensemble_obj.get("gender_model") or ensemble_obj.get("models")
        region_model = ensemble_obj.get("region_model")
        gender_encoder = ensemble_obj.get("le_gender") or ensemble_obj.get("gender_encoder")
        region_encoder = ensemble_obj.get("le_region") or ensemble_obj.get("region_encoder")
        
        # Check if models are loaded
        models_loaded = all([
            gender_model is not None,
            region_model is not None,
            gender_encoder is not None,
            region_encoder is not None
        ])
        
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
            print(f"Available keys in pickle file: {list(ensemble_obj.keys())}")
            print(f"Please ensure the models and encoders are properly saved to {MODEL_PATH}")
        else:
            print("✓ Models loaded successfully!")
            
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to load models: {e}")

# Load models when module is imported
load_models()

