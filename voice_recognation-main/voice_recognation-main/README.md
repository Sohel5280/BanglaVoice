## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd voice_recognation
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure you have the model file:**
   - Place `bangla_speech_models.pkl` in the project root directory
   - The file should contain: `gender_model`, `region_model`, `le_gender`, `le_region`

5. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

6. **Create a superuser (optional, for admin access):**
   ```bash
   python manage.py createsuperuser
   ```

## Running the Application

**Development server:**
```bash
python manage.py runserver
```

Then open your browser and navigate to:
```
http://localhost:8000
```

## Project Structure

```
voice_recognation/
├── manage.py
├── bangla_voice/          # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── voice_recognition/     # Main Django app
│   ├── views.py           # View functions
│   ├── urls.py            # URL routing
│   ├── model_loader.py    # ML model loading
│   └── templates/         # HTML templates
├── audio_utils.py         # Audio feature extraction
├── bangla_speech_models.pkl  # ML models (required)
└── requirements.txt       # Python dependencies
```

## API Endpoints

- `GET /` - Main page with UI
- `POST /predict` - Upload audio file and get predictions
- `GET /health` - Health check and model status
- `GET /api/info` - API information

## Deployment

### For Production:

1. **Update settings.py:**
   - Set `DEBUG = False`
   - Update `ALLOWED_HOSTS` with your domain
   - Set a secure `SECRET_KEY`
   - Configure proper database (PostgreSQL recommended)

2. **Collect static files:**
   ```bash
   python manage.py collectstatic
   ```

3. **Use a production WSGI server:**
   - Gunicorn: `gunicorn bangla_voice.wsgi:application`
   - Or deploy to platforms like:
     - Heroku
     - Railway
     - DigitalOcean
     - AWS Elastic Beanstalk
     - PythonAnywhere

## Requirements

- Python 3.8+
- Django 4.2+
- NumPy
- scikit-learn
- librosa
- soundfile

## License

This project is for educational and research purposes.

## Notes

- The application requires a properly trained model file (`bangla_speech_models.pkl`)
- Supported audio formats: WAV, MP3, OGG, FLAC
- Maximum file size: 10MB (configurable in settings.py)

