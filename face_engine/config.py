"""Central config shared by all face_engine scripts (kept independent of
Django settings so these scripts can also be run standalone from the CLI)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / 'media' / 'dataset'          # raw captured face crops
MODEL_DIR = BASE_DIR / 'face_engine' / 'trained_model'  # trained KNN + encoder
KNN_MODEL_PATH = MODEL_DIR / 'knn_classifier.pkl'
LABEL_ENCODER_PATH = MODEL_DIR / 'label_encoder.pkl'
EMBEDDING_CACHE_PATH = MODEL_DIR / 'embeddings_cache.pkl'

FACE_SIZE = (160, 160)                 # FaceNet's expected input size
SAMPLES_PER_STUDENT = 60                # frames captured per student on enrollment
RECOGNITION_CONFIDENCE_THRESHOLD = 0.75  # min vote-ratio from KNN to accept a match
DETECTION_CONFIDENCE_THRESHOLD = 0.90    # MTCNN face-detection confidence

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)
