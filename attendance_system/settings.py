"""
Django settings for the Smart Attendance System.
FaceNet + OpenCV + TensorFlow are used inside the `face_engine` package;
this file only wires up Django itself (SQLite DB, apps, templates, static/media).
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Security -----------------------------------------------------------
SECRET_KEY = 'change-this-secret-key-in-production'
DEBUG = True
ALLOWED_HOSTS = ['*']

# --- Apps -----------------------------------------------------------------
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'attendance',                      # our app
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'attendance_system.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'attendance' / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'attendance_system.wsgi.application'

# --- Database: SQLite (as required) ---------------------------------------
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# --- Face-engine specific settings ----------------------------------------
# Where captured face crops are stored per-student before training:
DATASET_DIR = MEDIA_ROOT / 'dataset'
# Where the trained KNN model + label encoder are stored after training:
MODEL_DIR = BASE_DIR / 'face_engine' / 'trained_model'
KNN_MODEL_PATH = MODEL_DIR / 'knn_classifier.pkl'
LABEL_ENCODER_PATH = MODEL_DIR / 'label_encoder.pkl'
# Minimum softmax-style confidence (KNN neighbor vote ratio) to accept a match
RECOGNITION_CONFIDENCE_THRESHOLD = 0.75
# How many face samples to capture per student during enrollment
SAMPLES_PER_STUDENT = 60

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
