"""Recognition with KNN confidence and an absolute FaceNet distance gate."""
import pickle
import cv2
import numpy as np
from face_engine import config
from face_engine.face_utils import detect_faces, crop_and_align, get_embedding

_bundle = None
_model_version = None


def _load_model():
    global _bundle, _model_version
    if not config.MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError('Train the recognition model from Enrollment before starting the camera.')
    version = config.MODEL_BUNDLE_PATH.stat().st_mtime_ns
    if _bundle is None or version != _model_version:
        # Only load locally generated, trusted artifacts; pickle is executable.
        with open(config.MODEL_BUNDLE_PATH, 'rb') as handle:
            bundle = pickle.load(handle)
        _bundle, _model_version = bundle, version
    return _bundle['knn'], _bundle['encoder']


def classify_embedding(embedding, knn, encoder):
    embedding = np.asarray(embedding).reshape(1, -1)
    probabilities = knn.predict_proba(embedding)[0]
    index = int(probabilities.argmax())
    predicted = knn.classes_[index]
    confidence = float(probabilities[index])
    # Check distance to the predicted identity, rather than KNN vote agreement alone.
    distances, indices = knn.kneighbors(embedding)
    matching = knn._y[indices[0]] == index
    distance = float(distances[0][matching].min()) if matching.any() else float('inf')
    accepted = confidence >= config.RECOGNITION_CONFIDENCE_THRESHOLD and distance <= config.MAX_EMBEDDING_DISTANCE
    return {
        'student_id': str(encoder.inverse_transform([predicted])[0]) if accepted else None,
        'confidence': confidence, 'distance': distance,
    }


def recognize_face_in_frame(frame_bgr):
    knn, encoder = _load_model()
    results = []
    for detection in detect_faces(frame_bgr):
        crop = crop_and_align(frame_bgr, detection['box'])
        if crop is not None:
            result = classify_embedding(get_embedding(crop), knn, encoder)
            results.append({'box': detection['box'], **result})
    return results


if __name__ == '__main__':
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    try:
        if not cap.isOpened():
            raise RuntimeError('Could not open webcam.')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for result in recognize_face_in_frame(frame):
                x, y, w, h = result['box']
                color = (0, 200, 0) if result['student_id'] else (0, 0, 220)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, result['student_id'] or 'Unknown', (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow('Recognition demo - q to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
