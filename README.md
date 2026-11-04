# FaceTrack — FaceNet-Based Smart Attendance

A local, staff-operated attendance application using **Python, FaceNet, OpenCV,
TensorFlow, Django, KNN (scikit-learn), and SQLite**.

## Features

- Staff authentication and CSRF-protected enrollment/training/camera initiation.
- Student registration with validated identifiers and email addresses.
- OpenCV webcam capture with MTCNN detection, single-person checks and 160×160 crops.
- Pretrained FaceNet 512-dimensional embeddings, L2 normalization, batched extraction,
  and distance-weighted KNN identity verification.
- Unknown-face rejection using both KNN score and absolute embedding distance.
- Automatic attendance with a database constraint allowing one row per student per day.
- Responsive dashboard, daily attendance trends, date filters, CSV and PDF exports.
- Saved training metadata: actual sample counts, student counts, validation accuracy,
  training time and model configuration.
- Atomic model bundle replacement and automatic recognition-model reload after training.

## Requirements

- **64-bit Python 3.11** (the pinned TensorFlow 2.15 stack supports Python 3.9–3.11).
- A webcam and a desktop display on the computer running Django.
- Internet for installing dependencies, downloading pretrained FaceNet weights on first
  embedding extraction, and Bootstrap styling from its CDN.
- CPU execution is supported. TensorFlow 2.15 does not support native Windows CUDA GPU
  execution; use WSL2 if GPU execution is required. See the
  [TensorFlow installation guide](https://www.tensorflow.org/install/pip) and
  [tested build matrix](https://www.tensorflow.org/install/source_windows).

## Run on Windows (PowerShell)

Open PowerShell in the project directory:

```powershell
cd D:\FAS
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe manage.py migrate
.\venv\Scripts\python.exe manage.py createsuperuser
.\venv\Scripts\python.exe manage.py check
.\venv\Scripts\python.exe manage.py runserver --noreload
```

If `venv` already exists with Python 3.11, skip its creation. These commands use the
virtual environment directly, so PowerShell activation policies do not affect setup.
Keep this terminal open. Visit **http://127.0.0.1:8000/** and sign in with the superuser
credentials you just created. Stop the server with **Ctrl+C**.

You can alternatively activate the environment with `.\venv\Scripts\Activate.ps1`
and then use `python` instead of `.\venv\Scripts\python.exe`.

## Run on Linux

Use Python 3.11 and a desktop session with webcam access:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver --noreload
```

## Enrollment → training → attendance

1. Open **Students & enrollment**. Add each student with a unique identifier such as
   `STU001`, their name, department and optional email. Identifiers accept letters,
   numbers, hyphens and underscores.
2. Obtain consent before capturing biometric data. Click **Capture faces**. An OpenCV
   window opens on the **server computer**, using its webcam. Keep exactly one face
   in view and slowly vary pose. Each run adds up to 60 samples; press **q** to cancel.
   Capture times out after about 150 seconds. Partial samples remain available and a
   retry adds more without overwriting previous captures.
3. Repeat for at least **two students**, with at least **five readable samples each**.
   Ten students with 60 samples each produce a dataset of roughly 600 samples.
4. Click **Train model** and wait. The first run downloads FaceNet weights. Training
   fits KNN over FaceNet embeddings; it does not retrain the FaceNet neural network.
   Only registered students represented in the trained artifact become enrolled.
   CLI equivalent:

   ```powershell
   .\venv\Scripts\python.exe manage.py train_recognizer
   ```

5. Open **Live attendance**, then click **Start attendance session**. Stand in front
   of the server webcam. A green box with your name and “Present” confirms check-in.
   Unknown or insufficiently close faces remain red and do not create records.
6. Click **Stop camera** before starting capture or another live session. Run a single
   local server process: camera ownership locks coordinate threads in that process.
7. Open **Reports**, choose a date range, then export **CSV** or **PDF**. The dashboard
   shows current enrolled-student totals, today's rate and seven days of activity.

To add another student, register, capture and retrain. Existing legacy model files
(`knn_classifier.pkl` / `label_encoder.pkl`) must be regenerated: this version uses
one normalized-embedding bundle, `face_engine/trained_model/recognizer.pkl`.

## Settings

Face-engine constants live in `face_engine/config.py`.

| Setting | Default | Purpose |
|---|---|---|
| `SAMPLES_PER_STUDENT` | 60 | Number of samples added by a capture run |
| `MIN_SAMPLES_PER_STUDENT` | 5 | Minimum readable images per identity for training |
| `EMBEDDING_BATCH_SIZE` | 32 | Bound embedding inference batches |
| `RECOGNITION_CONFIDENCE_THRESHOLD` | 0.75 | Minimum distance-weighted KNN score |
| `MAX_EMBEDDING_DISTANCE` | 0.8 | Maximum distance between normalized embeddings |
| `DETECTION_CONFIDENCE_THRESHOLD` | 0.90 | Minimum MTCNN detection confidence |

Environment variables, set **before** starting Django:

```powershell
$env:CAMERA_INDEX = '0'             # Try 1 if a different camera is needed
$env:FACE_MAX_DISTANCE = '0.8'       # Calibrate with your known/unknown validation set
$env:DJANGO_DEBUG = '1'             # Local development
$env:DJANGO_ALLOWED_HOSTS = 'localhost,127.0.0.1,[::1]'
```

For a deployment, set a random `DJANGO_SECRET_KEY`, disable debug, configure allowed
hosts and HTTPS, run Django deployment checks, and use a supported production server.
The provided pinned dependencies are for reproducing the local demo and require a
security/compatibility update review before deployment. This webcam workflow is
intended for a **local desktop kiosk**, not a headless/cloud multiworker deployment.
A browser on another computer still uses the server's webcam; browser-camera upload
is not implemented. Staff accounts are created through `/admin/` with `is_staff` set.

## Validation and performance claims

The requested **500+ samples**, **98% accuracy**, **80% less manual attendance time**,
and **70% greater processing efficiency** are benchmark targets, not results verified
by this repository. The application does not fabricate those numbers.

Training records real dataset counts and image-split accuracy in
`face_engine/trained_model/training_report.json`. Adjacent frames from the same capture
session can make a random image split optimistic. For a defensible recognition
benchmark, keep a separately captured session out of training, include unknown
subjects, and report false accepts/rejects as well as accepted-match accuracy.
Calibrate the distance threshold on that data. A KNN score is not a calibrated
probability. Measure manual and automated check-in times on the same cohort before
claiming time or efficiency gains.

This application has **no liveness detection**; a photo or replay may spoof it.
Use supervised capture/check-in where that matters. Student records and face crops
are staff-only; raw `media/` files are deliberately not served by Django. Restrict
filesystem access, define consent/retention policies, and back up the SQLite database
and biometric files appropriately. Only load model pickle files created and trusted
locally.

## Tests

```powershell
.\venv\Scripts\python.exe manage.py test
.\venv\Scripts\python.exe manage.py makemigrations --check --dry-run
```

Tests cover staff access, CSRF, registration validation, POST-only actions, invalid
date filters, CSV/PDF export, enrollment status, duplicate attendance, stream cleanup,
unknown-face rejection, bounding boxes and small-dataset training. They use synthetic
embeddings and mocked camera inputs, so they do not download weights or measure real
recognition accuracy. A real webcam/FaceNet smoke test is still required locally.

## Troubleshooting

- **No TensorFlow wheel / installation fails:** use 64-bit Python 3.11 and the venv
  commands above, rather than Python 3.12+ with these pinned dependencies.
- **TensorFlow DLL load failure on Windows:** install the Microsoft Visual C++
  Redistributable described in the TensorFlow installation guide.
- **Camera will not open:** close Teams/Zoom/other camera apps, enable OS camera
  permissions, try `CAMERA_INDEX=1`, and stop any existing attendance stream.
- **Training needs more samples:** capture at least two students and five readable
  JPG images for each. Prefer many varied samples across separate sessions.
- **First training is slow / download fails:** allow pretrained-weight download,
  check the server terminal and internet access, then retry.
- **Camera area shows an error:** stop the session, check the terminal, ensure the
  model is trained and restart. A session URL is single-use; start a new session.
- **Too many unknowns / false matches:** improve lighting and enrollment diversity,
  retrain, and calibrate the confidence and distance thresholds on held-out subjects.

## Project layout

```text
attendance/                  Django models, forms, staff views and tests
  templates/attendance/      Dashboard, enrollment, camera, reports, login
  static/attendance/         Responsive workspace stylesheet
  management/commands/       train_recognizer
attendance_system/           Django settings and URL routing
face_engine/                 Detection, FaceNet embeddings, KNN, capture
  trained_model/             Local model bundle and validation metadata (ignored)
media/dataset/<student_id>/   Private captured face crops (ignored)
db.sqlite3                   SQLite database (ignored)
```
