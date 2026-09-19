# FaceNet-Based Smart Attendance System

Django + OpenCV + MTCNN + FaceNet (TensorFlow/Keras) + scikit-learn KNN + SQLite.

## How the pipeline fits together

```
 [Webcam] --> MTCNN face detection --> crop & resize (160x160)
     --> FaceNet embedding (512-D vector, TensorFlow/Keras)
     --> KNN classifier (scikit-learn) --> student_id + confidence
     --> Django ORM writes an Attendance row to SQLite (once/day/student)
```

- **face_engine/** — pure Python, no Django imports. Detection, embedding,
  training and recognition logic live here so they can be run/tested from
  the command line independently of the web server.
- **attendance/** — the Django app: models (Student, Attendance), views
  (dashboard, enrollment, live camera feed, reports/CSV/PDF export),
  templates (Bootstrap + Chart.js).
- **db.sqlite3** — created automatically by Django migrations.

## 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

> First run of `keras-facenet` downloads pretrained FaceNet weights
> automatically and caches them locally — you need internet access once.

## 2. Set up the database

```bash
python manage.py makemigrations attendance
python manage.py migrate
python manage.py createsuperuser   # for /admin/
```

## 3. Run the server

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000/`.

## 4. Enroll students (build the 500+ sample dataset)

1. Go to **Register / Enroll**, add each student (ID + name).
2. Click **Capture Faces** next to their row — a webcam window opens on
   the server machine and saves ~60 face crops to
   `media/dataset/<student_id>/`. Repeat for every student (e.g. 10
   students x 60 samples ≈ 600 images, matching "500+ facial samples").
   You can raise `SAMPLES_PER_STUDENT` in `face_engine/config.py`.
3. Once all students are captured, click **Train Model**. This:
   - loads every image in `media/dataset/`
   - computes FaceNet embeddings for all of them in a batch
   - fits a `KNeighborsClassifier` on (embedding -> student_id)
   - saves `face_engine/trained_model/knn_classifier.pkl` and
     `label_encoder.pkl`
   - prints a held-out validation accuracy (with clean, well-lit,
     varied-pose samples this typically lands in the high-90s%, matching
     the "98% recognition accuracy" target)

   You can also do this from the CLI: `python manage.py train_recognizer`

## 5. Mark attendance (contactless)

Go to **Mark Attendance**. The live camera feed detects faces, classifies
them with the trained KNN model, and — the first time a recognized student
is seen that calendar day — writes an `Attendance` row automatically. A
green box + name means "recognized & marked"; a red box means "no
confident match" (confidence below `RECOGNITION_CONFIDENCE_THRESHOLD`,
default 0.75), so unknown faces are never marked present.

## 6. Reports & analytics

**Reports** page: filter by date range, see a per-student attendance
summary, and export the full record set as **CSV** or a formatted **PDF**
(via reportlab). The **Dashboard** shows today's present/absent counts,
attendance rate, and a 7-day trend chart (Chart.js).

## Tuning notes

| Setting | File | Effect |
|---|---|---|
| `SAMPLES_PER_STUDENT` | face_engine/config.py | more samples per person -> more robust embeddings, slower capture |
| `RECOGNITION_CONFIDENCE_THRESHOLD` | face_engine/config.py | higher -> fewer false accepts, more "Unknown" results |
| `DETECTION_CONFIDENCE_THRESHOLD` | face_engine/config.py | MTCNN's minimum confidence to count something as a face at all |
| `n_neighbors` | face_engine/train_model.py | KNN's k; auto-capped to the smallest class size |

## Why KNN on top of FaceNet (not FaceNet alone)

FaceNet's Inception-ResNet-v1 backbone is trained with triplet loss so that
embeddings of the *same* person end up close together in 512-D space and
different people end up far apart — but the network itself doesn't output
"student names." KNN is the lightweight classifier layered on top: enrolling
a new student only means adding their embeddings to the KNN's training set
and refitting KNN (milliseconds), never retraining the deep network.
