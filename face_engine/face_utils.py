"""
face_utils.py
=============
Shared, reusable building blocks for the whole face pipeline:

  1. detect_faces()      -> MTCNN face detector          (OpenCV frame in)
  2. get_embedding()     -> FaceNet 128-D embedding       (face crop in)
  3. FaceNetEngine        -> a small singleton wrapper so the (heavy) FaceNet
                             and MTCNN models are loaded into memory ONCE,
                             not on every single frame/request.

Every other script (capture_faces.py, train_model.py, recognize_attendance.py,
and the Django streaming view) imports this module instead of re-loading
TensorFlow models repeatedly, which is the single biggest performance trap
in real-time face recognition apps.
"""
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

from face_engine import config


class FaceNetEngine:
    """Lazily-initialised singleton holding the MTCNN detector and the
    pretrained FaceNet embedding model (Inception-ResNet-v1, TensorFlow/Keras
    backend, provided by the `keras-facenet` package)."""

    _detector = None
    _embedder = None

    @classmethod
    def detector(cls):
        if cls._detector is None:
            cls._detector = MTCNN()
        return cls._detector

    @classmethod
    def embedder(cls):
        if cls._embedder is None:
            # keras-facenet downloads/loads the pretrained FaceNet weights
            # the first time this runs and caches them locally afterwards.
            cls._embedder = FaceNet()
        return cls._embedder


def detect_faces(frame_bgr):
    """
    Run MTCNN on a BGR OpenCV frame.
    Returns a list of dicts: {'box': (x, y, w, h), 'confidence': float}
    Boxes are clipped to the frame so downstream cropping never goes negative.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = FaceNetEngine.detector().detect_faces(rgb)

    h_frame, w_frame = frame_bgr.shape[:2]
    faces = []
    for det in detections:
        if det['confidence'] < config.DETECTION_CONFIDENCE_THRESHOLD:
            continue
        x, y, w, h = det['box']
        x, y = max(0, x), max(0, y)
        w, h = min(w, w_frame - x), min(h, h_frame - y)
        faces.append({'box': (x, y, w, h), 'confidence': det['confidence']})
    return faces


def crop_and_align(frame_bgr, box):
    """Crop the detected face out of the frame and resize to FaceNet's
    expected 160x160 input. (Simple crop+resize; MTCNN's 5-point landmarks
    could be used for a full affine alignment if higher accuracy is needed.)"""
    x, y, w, h = box
    face = frame_bgr[y:y + h, x:x + w]
    if face.size == 0:
        return None
    face = cv2.resize(face, config.FACE_SIZE)
    return face


def get_embedding(face_bgr):
    """
    Convert a single aligned 160x160 BGR face crop into FaceNet's 128-D
    embedding vector. This embedding is what the KNN classifier is trained
    and queried on -- NOT the raw pixels.
    """
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    # keras_facenet expects a *batch* of images: shape (n, 160, 160, 3)
    embeddings = FaceNetEngine.embedder().embeddings([rgb])
    return embeddings[0]  # -> np.ndarray shape (512,) for this package's model


def get_embeddings_batch(face_list_bgr):
    """Vectorised version of get_embedding for training (much faster than
    calling get_embedding() in a Python loop over hundreds of images)."""
    rgb_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in face_list_bgr]
    return FaceNetEngine.embedder().embeddings(rgb_list)
