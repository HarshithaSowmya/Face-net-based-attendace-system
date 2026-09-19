"""
recognize_attendance.py
========================
Step 3 of the pipeline: RECOGNITION + shared logic used by the live Django
video feed (attendance/views.py) as well as this standalone CLI script.

Given a single video frame:
  1. Detect face(s) with MTCNN.
  2. Get each face's FaceNet embedding.
  3. Ask the trained KNN model to classify the embedding AND compute a
     confidence score (fraction of the k neighbors that agree on the label).
  4. If confidence >= RECOGNITION_CONFIDENCE_THRESHOLD, accept the match.

This module intentionally has NO Django imports, so it can be unit-tested
or run from the command line independently of the web server. The Django
view (attendance/views.py::gen_frames) imports `recognize_face_in_frame`
and handles the actual "write an Attendance row to SQLite" part itself,
since that requires Django's ORM.
"""
import pickle
import sys
from pathlib import Path

# See the comment in capture_faces.py: this keeps the script runnable both
# as `python face_engine/recognize_attendance.py` and as a normal import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from face_engine import config
from face_engine.face_utils import detect_faces, crop_and_align, get_embedding

_knn = None
_encoder = None


def _load_model():
    global _knn, _encoder
    if _knn is None or _encoder is None:
        if not config.KNN_MODEL_PATH.exists():
            raise FileNotFoundError(
                "No trained model found. Run face_engine/train_model.py first."
            )
        with open(config.KNN_MODEL_PATH, 'rb') as f:
            _knn = pickle.load(f)
        with open(config.LABEL_ENCODER_PATH, 'rb') as f:
            _encoder = pickle.load(f)
    return _knn, _encoder


def recognize_face_in_frame(frame_bgr):
    """
    Runs the full detect -> embed -> classify pipeline on one frame.

    Returns a list of results, one per detected face:
        {
            'box': (x, y, w, h),
            'student_id': str or None,   # None if below confidence threshold
            'confidence': float,          # 0.0 - 1.0, KNN neighbor-vote ratio
        }
    """
    knn, encoder = _load_model()
    results = []

    for det in detect_faces(frame_bgr):
        crop = crop_and_align(frame_bgr, det['box'])
        if crop is None:
            continue
        embedding = get_embedding(crop).reshape(1, -1)

        # Look at the k nearest neighbors and turn their vote into a
        # confidence score, rather than trusting predict() blindly.
        neighbor_dists, neighbor_idx = knn.kneighbors(embedding)
        neighbor_labels = knn._y[neighbor_idx[0]]
        predicted_label = np.bincount(neighbor_labels).argmax()
        confidence = float(np.mean(neighbor_labels == predicted_label))

        student_id = None
        if confidence >= config.RECOGNITION_CONFIDENCE_THRESHOLD:
            student_id = encoder.inverse_transform([predicted_label])[0]

        results.append({
            'box': det['box'],
            'student_id': student_id,
            'confidence': confidence,
        })

    return results


if __name__ == '__main__':
    # Simple standalone demo: open webcam, draw boxes + names, no DB writes.
    cap = cv2.VideoCapture(0)
    print("[recognize] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for r in recognize_face_in_frame(frame):
            x, y, w, h = r['box']
            label = r['student_id'] or "Unknown"
            color = (0, 255, 0) if r['student_id'] else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} ({r['confidence']:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow('Recognition demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
