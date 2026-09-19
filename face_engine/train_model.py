"""
train_model.py
===============
Step 2 of the pipeline: TRAINING.

Reads every captured face image under media/dataset/<student_id>/*.jpg,
converts each one into a 512-D FaceNet embedding, and fits a K-Nearest
Neighbors classifier (scikit-learn) on (embeddings -> student_id labels).

Why KNN on top of FaceNet embeddings (this matches the resume bullet
"KNN-based identity verification"):
  - FaceNet's job is only to produce embeddings where the same person's
    faces cluster tightly together and different people's faces are far
    apart (trained with triplet loss).
  - KNN is then a cheap, interpretable classifier on top of that embedding
    space: "whose embeddings, among all enrolled students, are closest to
    this new face?" It requires no retraining of the deep network itself
    when a new student enrolls -- you only refit the lightweight KNN.

Usage:
    python face_engine/train_model.py
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

# See the comment in capture_faces.py: this keeps the script runnable both
# as `python face_engine/train_model.py` and as a normal import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from face_engine import config
from face_engine.face_utils import get_embeddings_batch


def load_dataset():
    """Walk media/dataset/<student_id>/*.jpg and return (images, labels)."""
    images, labels = [], []
    for student_dir in sorted(config.DATASET_DIR.iterdir()):
        if not student_dir.is_dir():
            continue
        student_id = student_dir.name
        for img_path in student_dir.glob('*.jpg'):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                labels.append(student_id)
    return images, labels


def train():
    print("[train] Loading dataset...")
    images, labels = load_dataset()
    if len(set(labels)) < 2:
        raise RuntimeError(
            "Need at least 2 enrolled students with captured samples before training. "
            "Run capture_faces.py for each student first."
        )

    counts = defaultdict(int)
    for l in labels:
        counts[l] += 1
    print(f"[train] {len(images)} total images across {len(counts)} students: {dict(counts)}")

    print("[train] Computing FaceNet embeddings (this loads TensorFlow, may take a moment)...")
    embeddings = get_embeddings_batch(images)
    embeddings = np.asarray(embeddings)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42, stratify=y
    )

    # n_neighbors=5 is a reasonable default; lower it if any student has
    # fewer than 5 samples.
    n_neighbors = min(5, min(counts.values()))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean', weights='distance')
    knn.fit(X_train, y_train)

    val_preds = knn.predict(X_test)
    acc = accuracy_score(y_test, val_preds)
    print(f"[train] Validation accuracy: {acc * 100:.2f}%  (n_neighbors={n_neighbors})")

    # Refit on the FULL dataset for the deployed model (common practice once
    # you've validated the approach on the held-out split above).
    knn.fit(embeddings, y)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.KNN_MODEL_PATH, 'wb') as f:
        pickle.dump(knn, f)
    with open(config.LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(encoder, f)

    print(f"[train] Saved KNN model -> {config.KNN_MODEL_PATH}")
    print(f"[train] Saved label encoder -> {config.LABEL_ENCODER_PATH}")
    return acc


if __name__ == '__main__':
    train()
