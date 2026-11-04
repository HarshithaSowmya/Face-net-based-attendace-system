# FaceTrack — FaceNet-Based Smart Attendance

A staff-operated attendance website using **Python 3.11, Django 5.2 LTS, FaceNet,
TensorFlow/Keras, OpenCV, MTCNN, scikit-learn KNN, and SQLite**.

## Run locally in VS Code (PowerShell)

Open the project folder and a PowerShell terminal:

```powershell
cd D:\FAS
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe manage.py migrate
.\venv\Scripts\python.exe manage.py createsuperuser
.\venv\Scripts\python.exe manage.py runserver --noreload
```

Skip creating `venv` if it already exists. Open **http://127.0.0.1:8000/**.
Use port 8001 in the runserver command if another server already uses 8000.
Keep the terminal open; Ctrl+C stops Django. A forgotten password can be reset with
`python manage.py changepassword YOUR_USERNAME` using the virtual environment.

## Deploy on Render

[Deploy this repository to Render](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2FHarshithaSowmya%2FFace-net-based-attendace-system)

The included `render.yaml` defines a **paid 1 CPU / 2 GB web service with a 5 GB disk**.
Review Render's current compute and disk prices before approving creation:
[compute plans](https://render.com/docs/compute-plans),
[persistent disks](https://render.com/docs/disks), [pricing](https://render.com/pricing).
TensorFlow is memory intensive; the selected plan provides 2 GB RAM as a starting
point for a small classroom/demo. Actual memory use and CPU latency need monitoring.
Free services cannot retain this app's SQLite database and biometric files across
restarts. Do not deploy the current SQLite design without its persistent disk.

1. Sign in to [Render](https://dashboard.render.com/).
2. Choose **New → Blueprint**, connect GitHub, and select
   `HarshithaSowmya/Face-net-based-attendace-system` on branch `main`.
3. Review the service and disk from `render.yaml`.
4. Provide these **secret environment values** when prompted:

   | Key | Value |
   |---|---|
   | `DJANGO_SUPERUSER_USERNAME` | Your new online administrator username |
   | `DJANGO_SUPERUSER_EMAIL` | Your email address |
   | `DJANGO_SUPERUSER_PASSWORD` | A unique strong password for this online app |

   Render generates `DJANGO_SECRET_KEY` automatically. Do not put passwords in GitHub
   or reuse the local demo password. Your local accounts, face samples and attendance
   database are **not uploaded**: this creates a separate, initially empty online app.
5. Review the displayed recurring cost, then deploy the Blueprint.
6. Wait for the service to become **Live**. Open the service's actual HTTPS URL and
   sign in with the credentials entered in step 4.
7. Remove `DJANGO_SUPERUSER_PASSWORD` from Render's environment after the first
   successful login. Future starts leave existing users and passwords unchanged.
8. Enroll students and train the model on the online service. Allow the first
   FaceNet weight download to finish; the model cache is persisted on the disk.

`.python-version` selects Python 3.11, including its latest available security patch.

No Render account, API key or billing information is stored in this repository.
GitHub source preparation alone does not mean a Render service is live. Automatic
deploys are disabled; deploy a reviewed commit manually from Render for later updates.

### Manual web-service configuration (alternative)

| Render field | Value |
|---|---|
| Runtime | Python |
| Branch | `main` |
| Build command | `bash build.sh` |
| Start command | `bash start.sh` |
| Health check | `/health/` |
| Instances | 1 |
| Plan | `1c-2g` (review current pricing) |
| Persistent disk | 5 GB mounted at `/var/data` |
| Python version | `.python-version` selects the latest Python 3.11 patch |
| `DJANGO_DEBUG` | `0` |
| `DATA_DIR` | `/var/data` |
| `FACENET_CACHE_DIR` | `/var/data/facenet-cache` |
| `DJANGO_SECRET_KEY` | A long random secret, set in Render |
| `DJANGO_SUPERUSER_*` | Initial administrator credentials as above |

`RENDER_EXTERNAL_HOSTNAME` automatically configures the allowed host and CSRF origin.
For a custom domain, also set `DJANGO_ALLOWED_HOSTS` and
`DJANGO_CSRF_TRUSTED_ORIGINS` (comma-separated; CSRF origins include `https://`).
Do not use `*` as an allowed host.

The build installs CPU TensorFlow and headless OpenCV, then collects static files.
Startup creates the data directory, applies migrations **after the disk is mounted**,
optionally bootstraps the administrator, and starts Gunicorn. A single worker with
four threads shares one inference/training lock; do not increase workers or instances
without replacing that coordination and the SQLite/filesystem storage architecture.
Training requests have a 600-second server timeout. Very large datasets require
background-job infrastructure; this configuration targets small datasets.

## Browser webcam workflow

The website uses **the visitor's browser camera**, on desktop or mobile. It never
tries to open a camera on Render. Camera permission requires **HTTPS or localhost**.

1. Sign in with a staff account. Administrators can create more users in `/admin/`,
   checking **Active** and **Staff status**. All staff can manage attendance and
   enrollment; superuser status additionally grants full Django admin permissions.
2. Open **Students & enrollment**. Add each student's unique ID, name, department,
   and optional email. IDs accept letters, numbers, hyphens and underscores.
3. Obtain the student's consent, click **Capture faces**, then **Start camera** and
   allow browser-camera access. Only start when the participant agrees to upload
   face snapshots to this attendance server.
4. Keep exactly one person in view and slowly vary pose and lighting. A capture run
   saves 60 face crops, then stops automatically. Partial captures remain available.
   Stop and start to add more if needed; the server caps each student at 600 crops.
5. Capture at least **two students**, with at least **five readable samples each**.
6. Return to enrollment and click **Train model**. Wait for the result. FaceNet
   generates normalized 512-D embeddings; KNN learns identities. This does not
   retrain the FaceNet neural network. Only students represented in the artifact
   become enrolled. CLI equivalent: `python manage.py train_recognizer`.
7. Open **Live attendance** and start the browser camera. Accepted matches show the
   student's name and **Present**. Attendance is saved once per student per local
   calendar day (Asia/Kolkata), enforced by the database.
8. Stop the camera or leave/hide the tab to release it. **Reports** supports validated
   date filters, per-student summaries, CSV and PDF export. The dashboard shows
   current enrolled-student totals, today's rate and seven days of activity.

One JPEG snapshot at a time is uploaded, up to 640×480 by the browser. Server
validation accepts JPEG/PNG images under 1 MB and at most 1280 pixels per edge.
Enrollment keeps only the face crop; check-in uploads are processed in memory and
not saved. Multiple faces are rejected for enrollment. Unknown or unenrolled
identities do not create attendance. If the engine is busy, clients retry.

The image title has been removed from the supplied campus background. It is served
with other public static assets; biometric media is never served as static content.

## Persistence and settings

| Data | Local default | Render |
|---|---|---|
| SQLite | `db.sqlite3` | `/var/data/db.sqlite3` |
| Private face crops | `media/dataset/` | `/var/data/media/dataset/` |
| Model + report | `face_engine/trained_model/` | `/var/data/face_engine/trained_model/` |
| FaceNet weights | `~/.keras-facenet` | `/var/data/facenet-cache` |
| Public collected assets | `staticfiles/` | Rebuilt each deploy |

`DATA_DIR` changes database, media and model storage together. `FACENET_CACHE_DIR`
controls pretrained-weight storage. Only disk-backed paths survive Render restarts
and redeploys. Keep database and biometric backups consistent. Never commit them.

Face-engine controls in `face_engine/config.py`:

| Setting | Default |
|---|---|
| Samples per capture | 60 |
| Minimum readable samples per identity | 5 |
| Embedding inference batch size | 32 |
| Minimum KNN score | 0.75 |
| Maximum normalized embedding distance | 0.8 |
| Minimum face detector confidence | 0.90 |

Set `FACE_MAX_DISTANCE` in the environment to override the distance threshold.
Calibrate this against your known and unknown validation subjects before relying on
matches. Existing legacy `knn_classifier.pkl` and `label_encoder.pkl` files must be
retrained into the current normalized-embedding `recognizer.pkl` bundle.

## Accuracy and security limits

The requested **500+ samples**, **98% recognition accuracy**, **80% less attendance
time** and **70% greater processing efficiency** are targets, not measured results.
Real sample counts and image-split accuracy are saved to `training_report.json`.
Adjacent frames from one capture session can inflate random-split accuracy. Evaluate
on independently captured sessions and unknown people; measure false accepts and
false rejects, and record comparable manual/automated timing before making claims.
KNN confidence is a score, not a calibrated probability.

There is **no liveness detection**: a photo or replay may spoof this system. Use
supervised attendance where appropriate. Cloud capture sends biometric images to
Render for processing/storage; apply consent, retention and access policies. The
site remains staff-only, with CSRF protection, secure cookies, HTTPS enforcement,
validated uploads and bounded inference concurrency. Only load trusted locally
created pickle models. FaceNet retains the TensorFlow/Keras 2.15 compatibility
stack; review ML-library advisories and plan its migration before use beyond a
controlled pilot. This repository is not a security certification.

## Tests and verification

```powershell
.\venv\Scripts\python.exe manage.py test
.\venv\Scripts\python.exe manage.py check
.\venv\Scripts\python.exe manage.py makemigrations --check --dry-run
.\venv\Scripts\python.exe manage.py collectstatic --noinput
```

Tests cover authentication/CSRF, safe identifiers, invalid/oversized uploads,
multiple-face enrollment, private crop persistence, enrollment state, bounded
concurrency, unknown-face rejection, one daily attendance row, date filters,
CSV/PDF export, administrator bootstrap and synthetic small-dataset training.
They do not measure real recognition accuracy or grant camera permissions.

For production checks, set `DJANGO_DEBUG=0` and a real `DJANGO_SECRET_KEY`, then run
`python manage.py check --deploy`. Use `DJANGO_ALLOWED_HOSTS` for the intended host.

## Troubleshooting

- **Build fails with no TensorFlow wheel:** use the declared Python 3.11 version.
- **Service runs out of memory:** review Render logs, reduce embedding batch size
  and concurrent usage, or choose a larger plan. Do not add workers for this design.
- **Camera unavailable:** use HTTPS, allow permission, close other camera apps and
  use a supported browser. The in-app browser may not expose camera support; try
  Chrome or Edge with the public HTTPS URL.
- **First inference/training slow:** allow weight download and TensorFlow startup.
  Keep the tab visible. Review logs if the operation fails.
- **Login fails online:** local accounts are separate. Use the Render bootstrap
  credentials, or use Render Shell to run `python manage.py changepassword USER`.
- **No matches:** capture more varied data, retrain and tune distance thresholds.
- **Data disappears:** verify the disk mount and `DATA_DIR=/var/data`; source-tree
  storage on Render is ephemeral.

## Layout

```text
attendance/                  Models, forms, staff pages, camera API and tests
  static/attendance/         CSS, browser-camera JS and campus background
  management/commands/       train_recognizer and bootstrap_admin
attendance_system/           Environment-based Django settings and routing
face_engine/                 Detection, embeddings, KNN and optional local CLI demos
render.yaml                  Render Blueprint with persistent disk
build.sh / start.sh          Production build and runtime initialization
requirements-common.txt      Shared web/ML dependencies
requirements.txt             Local desktop dependencies
requirements-render.txt      Headless Linux/CPU production dependencies
```
