from django.apps import AppConfig


class VoiceRecognitionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'voice_recognition'
    
    def ready(self):
        """Load models when Django starts."""
        import voice_recognition.model_loader

