"""
URL configuration for voice_recognition app.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict', views.predict_speaker, name='predict'),
    path('health', views.health_check, name='health'),
    path('api/info', views.api_info, name='api_info'),
]

