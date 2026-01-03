"""
Example script showing how to properly save models to bangla_speech_models.pkl

This is a template - adapt it to your actual training code.
Make sure you replace the placeholder code with your actual trained models and encoders.
"""

import pickle
from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier  # or whatever model you're using
# from sklearn.svm import SVC
# etc.

# Example structure - REPLACE WITH YOUR ACTUAL TRAINED MODELS
# After training your models, you should have something like:

# gender_model = your_trained_gender_model  # e.g., RandomForestClassifier(), SVC(), etc.
# region_model = your_trained_region_model
# gender_encoder = LabelEncoder()  # fitted with your gender labels
# region_encoder = LabelEncoder()  # fitted with your region labels

# IMPORTANT: Make sure all models and encoders are TRAINED/FITTED before saving!

# Example (commented out - uncomment and adapt):
"""
# After training:
# gender_model.fit(X_train, y_gender_train)
# region_model.fit(X_train, y_region_train)
# gender_encoder.fit(y_gender_train)
# region_encoder.fit(y_region_train)

# Then save:
save_dict = {
    "gender_model": gender_model,      # ACTUAL trained model object, not None!
    "region_model": region_model,      # ACTUAL trained model object, not None!
    "le_gender": gender_encoder,       # ACTUAL fitted encoder object, not None!
    "le_region": region_encoder       # ACTUAL fitted encoder object, not None!
}

with open("bangla_speech_models.pkl", "wb") as f:
    pickle.dump(save_dict, f)

print("Models saved successfully!")
"""

# If you're loading from joblib (common with sklearn):
"""
import joblib

# Save with joblib (alternative method):
save_dict = {
    "gender_model": gender_model,
    "region_model": region_model,
    "le_gender": gender_encoder,
    "le_region": region_encoder
}

joblib.dump(save_dict, "bangla_speech_models.pkl")
"""

# To verify your saved file:
"""
import pickle

with open("bangla_speech_models.pkl", "rb") as f:
    loaded = pickle.load(f)
    
for key, value in loaded.items():
    print(f"{key}: {type(value)} - {value is not None}")
"""

