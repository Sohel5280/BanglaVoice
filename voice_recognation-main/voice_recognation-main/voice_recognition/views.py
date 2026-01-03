"""
Django views for voice recognition app.
"""
import io
import os
import json
import tempfile
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.uploadedfile import InMemoryUploadedFile
import numpy as np
from audio_utils import extract_features_from_filelike
from .model_loader import gender_model, region_model, gender_encoder, region_encoder, models_loaded


def index(request):
    """Render the main page."""
    return render(request, 'voice_recognition/index.html')


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
            f"Please ensure the models and encoders are properly saved."
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


@csrf_exempt
@require_http_methods(["POST"])
def predict_speaker(request):
    """API endpoint for voice prediction."""
    if 'audio' not in request.FILES:
        return JsonResponse({'detail': 'No audio file provided'}, status=400)
    
    audio_file = request.FILES['audio']
    
    # Basic validation
    valid_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    file_extension = os.path.splitext(audio_file.name)[1].lower()
    
    if file_extension not in valid_extensions:
        return JsonResponse({
            'detail': 'Unsupported audio format. Please use WAV, MP3, OGG, or FLAC.'
        }, status=400)
    
    try:
        # Save uploaded file to temporary file
        # librosa.load() works better with actual file paths, especially for MP3
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            # Write uploaded file content to temp file
            for chunk in audio_file.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        
        try:
            # Extract features from temporary file
            # librosa.load() accepts file paths and handles MP3 better this way
            features = extract_features_from_filelike(temp_file_path)
            
            # Run prediction
            gender, region = predict(features)
            
            return JsonResponse({
                'gender': gender,
                'region': region,
                'file_name': audio_file.name,
            })
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass  # Ignore cleanup errors
        
    except Exception as e:
        return JsonResponse({
            'detail': f'Prediction failed: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint."""
    models_status = {
        'gender_model': gender_model is not None,
        'region_model': region_model is not None,
        'gender_encoder': gender_encoder is not None,
        'region_encoder': region_encoder is not None
    }
    all_loaded = all(models_status.values())
    
    return JsonResponse({
        'status': 'healthy' if all_loaded else 'degraded',
        'models_loaded': all_loaded,
        'components': models_status,
        'ready_for_predictions': all_loaded
    })


@require_http_methods(["GET"])
def api_info(request):
    """API information endpoint."""
    return JsonResponse({
        'message': 'Bangla Speech Gender & Region API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'Main page',
            '/predict': 'POST endpoint for audio prediction',
            '/health': 'Health check and model status',
            '/api/info': 'API information (this endpoint)'
        },
        'models_status': 'loaded' if models_loaded else 'not loaded'
    })

