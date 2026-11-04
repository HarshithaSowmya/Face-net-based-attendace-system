"""Train a KNN recognizer over normalized FaceNet embeddings."""
import json
import math
import os
import pickle
import uuid
from collections import Counter
from datetime import datetime, timezone

import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from face_engine import config
from face_engine.face_utils import get_embeddings_batch


def load_dataset(student_ids=None):
    images, labels = [], []
    if not config.DATASET_DIR.exists():
        return images, labels
    for directory in sorted(config.DATASET_DIR.iterdir()):
        if not directory.is_dir() or (student_ids is not None and directory.name not in student_ids):
            continue
        for path in sorted(directory.glob('*.jpg')):
            image = cv2.imread(str(path))
            if image is not None:
                images.append(cv2.resize(image, config.FACE_SIZE))
                labels.append(directory.name)
    return images, labels


def training_report():
    try:
        return json.loads(config.TRAINING_REPORT_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def train(student_ids=None):
    images, labels = load_dataset(student_ids)
    counts = Counter(labels)
    if len(counts) < 2:
        raise RuntimeError('Capture samples for at least two registered students before training.')
    if min(counts.values()) < config.MIN_SAMPLES_PER_STUDENT:
        raise RuntimeError(f'Each student needs at least {config.MIN_SAMPLES_PER_STUDENT} readable JPG samples. Counts: {dict(counts)}')
    print(f'[train] Computing embeddings for {len(images)} images across {len(counts)} students.')
    embeddings = np.concatenate([
        get_embeddings_batch(images[start:start + config.EMBEDDING_BATCH_SIZE])
        for start in range(0, len(images), config.EMBEDDING_BATCH_SIZE)
    ])
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    validation_size = max(len(counts), math.ceil(len(y) * 0.2))
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=validation_size, random_state=42, stratify=y,
    )
    k = min(5, min(Counter(y_train).values()))
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='distance')
    knn.fit(X_train, y_train)
    accuracy = float(accuracy_score(y_test, knn.predict(X_test)))
    knn.fit(embeddings, y)
    report = {
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'sample_count': len(images), 'student_count': len(counts),
        'samples_per_student': dict(counts), 'student_ids': list(encoder.classes_),
        'validation_accuracy': accuracy, 'validation_samples': len(y_test),
        'neighbors': k, 'max_embedding_distance': config.MAX_EMBEDDING_DISTANCE,
        'validation_note': 'Random image split; adjacent capture frames can inflate accuracy. Validate on separate sessions and unknown subjects.',
    }
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # A single atomic bundle prevents readers seeing a new model with an old encoder.
    temporary = config.MODEL_BUNDLE_PATH.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    with open(temporary, 'wb') as handle:
        pickle.dump({'knn': knn, 'encoder': encoder, 'report': report}, handle)
    os.replace(temporary, config.MODEL_BUNDLE_PATH)
    temporary_report = config.TRAINING_REPORT_PATH.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    temporary_report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    os.replace(temporary_report, config.TRAINING_REPORT_PATH)
    print(f'[train] Validation accuracy: {accuracy:.2%}; saved {config.MODEL_BUNDLE_PATH}')
    return accuracy


if __name__ == '__main__':
    train()
