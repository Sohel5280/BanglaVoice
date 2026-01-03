import io
import numpy as np
import librosa

def extract_features_from_filelike(file_like, sr=16000, n_mfcc=61):
    """
    Extract comprehensive audio features to match training pipeline.
    Returns 122 features to match the trained model expectations.
    
    Feature breakdown (122 total):
    - MFCC: 61 coefficients * 2 (mean + std) = 122 features
    
    Args:
        file_like: Can be a file path (string) or file-like object (BytesIO, etc.)
    """
    # librosa.load() works better with file paths, especially for MP3 files
    # If it's a string, treat it as a file path; otherwise use it as file-like object
    data, _ = librosa.load(file_like, sr=sr, mono=True)
    
    # Extract MFCC features (61 coefficients to get 122 features with mean+std)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    
    # Calculate mean and std over time axis
    mfcc_mean = mfcc.mean(axis=1)  # 61 features
    mfcc_std = mfcc.std(axis=1)     # 61 features
    
    # Concatenate mean and std
    features = np.concatenate([mfcc_mean, mfcc_std], axis=0)
    
    # Ensure we have exactly 122 features
    if len(features) < 122:
        # Pad with zeros if we have fewer features
        features = np.pad(features, (0, 122 - len(features)), mode='constant')
    elif len(features) > 122:
        # Truncate if we have more features
        features = features[:122]
    
    return features.reshape(1, -1)
