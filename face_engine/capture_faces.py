"""
capture_faces.py
================
Step 1 of the pipeline: ENROLLMENT.

Opens the webcam, runs MTCNN face detection on every frame, and saves
cropped 160x160 face images to  media/dataset/<student_id>/  until
SAMPLES_PER_STUDENT images have been captured. These raw crops are later
turned into embeddings + a trained KNN model by train_model.py.

Usage (standalone CLI):
    python face_engine/capture_faces.py --id STU001 --name "Asha Rao"

This is also invoked from the Django "Register Student" view via subprocess,
so it can be triggered from the web UI.
"""
import argparse
import sys
import time
from pathlib import Path

# Make sure the project root (which contains the face_engine/ package) is on
# sys.path, so this file works whether it's run as `python face_engine/capture_faces.py`
# (script mode, where Python only puts face_engine/ itself on sys.path) or as
# `python -m face_engine.capture_faces` (module mode). Without this, script
# mode raises: ModuleNotFoundError: No module named 'face_engine'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from face_engine import config
from face_engine.face_utils import detect_faces, crop_and_align


def capture_for_student(student_id: str, num_samples: int = config.SAMPLES_PER_STUDENT):
    save_dir = config.DATASET_DIR / student_id
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0).")

    count = 0
    print(f"[capture] Starting capture for '{student_id}'. Press 'q' to abort early.")
    last_save = 0
    while count < num_samples:
        ok, frame = cap.read()
        if not ok:
            continue

        faces = detect_faces(frame)
        for f in faces:
            x, y, w, h = f['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # throttle saving slightly so we capture varied poses, not
            # 60 near-identical frames in half a second
            if faces and (time.time() - last_save) > 0.15:
                crop = crop_and_align(frame, f['box'])
                if crop is not None:
                    out_path = save_dir / f"{student_id}_{count:03d}.jpg"
                    cv2.imwrite(str(out_path), crop)
                    count += 1
                    last_save = time.time()
            break  # only use the largest/first detected face per frame

        cv2.putText(frame, f"Captured: {count}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Enrollment - press q to stop', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[capture] Done. Saved {count} images to {save_dir}")
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Capture face samples for a student.")
    parser.add_argument('--id', required=True, help="Unique student ID (folder name)")
    parser.add_argument('--name', required=False, help="Student full name (for logging only)")
    parser.add_argument('--samples', type=int, default=config.SAMPLES_PER_STUDENT)
    args = parser.parse_args()
    capture_for_student(args.id, args.samples)
