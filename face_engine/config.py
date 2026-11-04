"""Central config shared by all face_engine scripts (kept independent of
Django settings so these scripts can also be run standalone from the CLI)."""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('DATA_DIR', str(BASE_DIR)))
DATASET_DIR = DATA_DIR / 'media' / 'dataset'          # raw captured face crops
MODEL_DIR = DATA_DIR / 'face_engine' / 'trained_model'  # trained KNN + encoder
KNN_MODEL_PATH = MODEL_DIR / 'knn_classifier.pkl'
LABEL_ENCODER_PATH = MODEL_DIR / 'label_encoder.pkl'
EMBEDDING_CACHE_PATH = MODEL_DIR / 'embeddings_cache.pkl'
MODEL_BUNDLE_PATH = MODEL_DIR / 'recognizer.pkl'
TRAINING_REPORT_PATH = MODEL_DIR / 'training_report.json'
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '0'))
MIN_SAMPLES_PER_STUDENT = 5
EMBEDDING_BATCH_SIZE = 32
# Euclidean distance between L2-normalized embeddings. Calibrate with local
# known and unknown subjects before relying on this threshold.
MAX_EMBEDDING_DISTANCE = float(os.environ.get('FACE_MAX_DISTANCE', '0.8'))

FACE_SIZE = (160, 160)                 # FaceNet's expected input size
SAMPLES_PER_STUDENT = 60                # frames captured per student on enrollment
RECOGNITION_CONFIDENCE_THRESHOLD = 0.75  # min vote-ratio from KNN to accept a match
DETECTION_CONFIDENCE_THRESHOLD = 0.90    # MTCNN face-detection confidence

FACENET_CACHE_DIR = Path(os.environ.get('FACENET_CACHE_DIR', str(Path('~/.keras-facenet').expanduser())))
