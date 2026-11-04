"""Capture single-person enrollment samples using the server's webcam."""
import argparse
import re
import time
import uuid
import cv2
from face_engine import config
from face_engine.face_utils import detect_faces, crop_and_align


def capture_for_student(student_id, num_samples=config.SAMPLES_PER_STUDENT):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,29}', student_id):
        raise ValueError('Student ID must contain only letters, numbers, hyphens and underscores.')
    if not 1 <= num_samples <= 500:
        raise ValueError('Samples must be between 1 and 500.')
    directory = config.DATASET_DIR / student_id
    directory.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    count, last_save = 0, 0
    session = uuid.uuid4().hex[:12]
    deadline = time.monotonic() + 150
    try:
        if not cap.isOpened():
            raise RuntimeError('Could not open webcam. Check camera permissions and CAMERA_INDEX.')
        while count < num_samples and time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera stopped delivering frames.')
            faces = detect_faces(frame)
            # Multiple people can contaminate the identity's enrollment data.
            if len(faces) == 1:
                box = faces[0]['box']
                crop = crop_and_align(frame, box)
                if crop is not None and time.monotonic() - last_save > 0.25:
                    if cv2.imwrite(str(directory / f'{session}_{count:03d}.jpg'), crop):
                        count += 1
                        last_save = time.monotonic()
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            hint = f'Captured {count}/{num_samples}' if len(faces) == 1 else 'Keep exactly one face in view'
            cv2.putText(frame, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            cv2.imshow('Enrollment - vary pose slowly; q to cancel', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f'[capture] Saved {count} samples to {directory}')
    if count < num_samples:
        raise RuntimeError(f'Capture incomplete: saved {count}/{num_samples}. Retry to add more samples.')
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture face samples for one consenting student.')
    parser.add_argument('--id', required=True)
    parser.add_argument('--samples', type=int, default=config.SAMPLES_PER_STUDENT)
    args = parser.parse_args()
    capture_for_student(args.id, args.samples)
