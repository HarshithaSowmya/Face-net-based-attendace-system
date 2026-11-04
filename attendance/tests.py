import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction
from django.test import TestCase, SimpleTestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from attendance.models import Student, Attendance
from face_engine import config, train_model
from face_engine.capture_faces import capture_for_student
from face_engine.face_utils import detect_faces
from face_engine.recognize_attendance import classify_embedding


@override_settings(STORAGES={
    'default': {'BACKEND': 'django.core.files.storage.FileSystemStorage'},
    'staticfiles': {'BACKEND': 'django.contrib.staticfiles.storage.StaticFilesStorage'},
})
class AttendanceWebTests(TestCase):
    def setUp(self):
        self.staff = get_user_model().objects.create_user('staff', password='test-password', is_staff=True)
        self.client.force_login(self.staff)
        self.student = Student.objects.create(student_id='STU001', name='Asha', is_enrolled=True)

    def test_staff_required_for_pages_exports_and_camera(self):
        self.client.logout()
        for name in ['dashboard', 'register', 'reports', 'mark_attendance', 'recognize_frame', 'export_csv', 'export_pdf']:
            self.assertEqual(self.client.get(reverse(f'attendance:{name}')).status_code, 302)
        user = get_user_model().objects.create_user('viewer', password='test-password')
        self.client.force_login(user)
        self.assertEqual(self.client.get('/').status_code, 302)

    def test_pages_render_without_loading_tensorflow(self):
        for name in ['dashboard', 'register', 'reports', 'mark_attendance']:
            self.assertEqual(self.client.get(reverse(f'attendance:{name}')).status_code, 200)

    def test_registration_validates_email_and_safe_identifier(self):
        for identifier, email in [('../escape', ''), ('STU002', 'invalid-email')]:
            response = self.client.post(reverse('attendance:register'), {'student_id': identifier, 'name': 'Test', 'email': email})
            self.assertEqual(response.status_code, 200)
        self.assertEqual(Student.objects.count(), 1)
        self.client.post(reverse('attendance:register'), {'student_id': 'STU002', 'name': 'Ravi'})
        self.assertTrue(Student.objects.filter(student_id='STU002').exists())

    def test_mutations_reject_get(self):
        self.assertEqual(self.client.get(reverse('attendance:run_training')).status_code, 405)
        self.assertEqual(self.client.get(reverse('attendance:run_capture', args=['STU001'])).status_code, 200)
        self.assertEqual(self.client.get(reverse('attendance:capture_frame', args=['STU001'])).status_code, 405)
        self.assertEqual(self.client.get(reverse('attendance:recognize_frame')).status_code, 405)

    def test_training_only_enrolls_students_in_artifact(self):
        pending = Student.objects.create(student_id='STU002', name='Pending')
        with patch('attendance.views.face_trainer.train', return_value=0.98), patch(
            'attendance.views.face_trainer.training_report', return_value={'student_ids': ['STU001']},
        ):
            self.client.post(reverse('attendance:run_training'))
        pending.refresh_from_db()
        self.assertFalse(pending.is_enrolled)

    def test_invalid_report_dates_handled_for_page_and_exports(self):
        for params in [{'start': 'invalid'}, {'start': '2026-11-04', 'end': '2026-01-01'}]:
            self.assertEqual(self.client.get(reverse('attendance:reports'), params).status_code, 200)
            for name in ['export_csv', 'export_pdf']:
                self.assertEqual(self.client.get(reverse(f'attendance:{name}'), params).status_code, 400)

    def test_exports_include_records_and_escape_csv_formulas(self):
        self.student.name = '=FORMULA()'
        self.student.save()
        Attendance.objects.create(student=self.student, confidence=0.95)
        csv = self.client.get(reverse('attendance:export_csv'))
        self.assertContains(csv, "'=FORMULA()")
        pdf = self.client.get(reverse('attendance:export_pdf'))
        self.assertTrue(pdf.content.startswith(b'%PDF'))

    def test_database_rejects_duplicate_daily_attendance(self):
        Attendance.objects.create(student=self.student, confidence=0.95)
        with self.assertRaises(IntegrityError), transaction.atomic():
            Attendance.objects.create(student=self.student, confidence=0.95)

    def frame_upload(self, size=(64, 64)):
        from io import BytesIO
        from PIL import Image
        from django.core.files.uploadedfile import SimpleUploadedFile
        buffer = BytesIO()
        Image.new('RGB', size, 'white').save(buffer, format='JPEG')
        return SimpleUploadedFile('frame.jpg', buffer.getvalue(), content_type='image/jpeg')

    def test_browser_recognition_marks_once_per_local_day(self):
        result = {'box': (0, 0, 30, 30), 'student_id': 'STU001', 'confidence': 0.95}
        with patch('attendance.views.face_config.MODEL_BUNDLE_PATH') as model, patch('attendance.views.recognize_face_in_frame', return_value=[result]):
            model.exists.return_value = True
            for _ in range(2):
                response = self.client.post(reverse('attendance:recognize_frame'), {'frame': self.frame_upload()})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['results'][0]['name'], 'Asha')
        self.assertEqual(Attendance.objects.count(), 1)
        self.assertEqual(Attendance.objects.get().date, timezone.localdate())

    def test_unknown_and_inactive_enrollment_do_not_mark(self):
        self.student.is_enrolled = False
        self.student.save()
        result = {'box': (0, 0, 30, 30), 'student_id': 'STU001', 'confidence': 0.95}
        with patch('attendance.views.face_config.MODEL_BUNDLE_PATH') as model, patch('attendance.views.recognize_face_in_frame', return_value=[result]):
            model.exists.return_value = True
            response = self.client.post(reverse('attendance:recognize_frame'), {'frame': self.frame_upload()})
            self.assertFalse(response.json()['results'][0]['recognized'])
        self.assertEqual(Attendance.objects.count(), 0)

    def test_capture_saves_private_crop_and_requires_retraining(self):
        with tempfile.TemporaryDirectory(dir=config.BASE_DIR) as directory, patch.object(config, 'DATASET_DIR', Path(directory)), patch('attendance.views.detect_faces', return_value=[{'box': (0, 0, 40, 40)}]):
            response = self.client.post(reverse('attendance:capture_frame', args=['STU001']), {'frame': self.frame_upload()})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()['saved'])
            self.assertEqual(len(list((Path(directory) / 'STU001').glob('*.jpg'))), 1)
        self.student.refresh_from_db()
        self.assertFalse(self.student.is_enrolled)

    def test_multiple_faces_are_not_saved(self):
        with patch('attendance.views.detect_faces', return_value=[{}, {}]):
            response = self.client.post(reverse('attendance:capture_frame', args=['STU001']), {'frame': self.frame_upload()})
            self.assertFalse(response.json()['saved'])

    def test_camera_rejects_missing_invalid_and_oversized_frames(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        url = reverse('attendance:capture_frame', args=['STU001'])
        self.assertEqual(self.client.post(url).status_code, 400)
        for frame in [SimpleUploadedFile('bad.jpg', b'not an image'), self.frame_upload((1281, 32)), SimpleUploadedFile('huge.jpg', b'x' * (1024 * 1024 + 1))]:
            self.assertEqual(self.client.post(url, {'frame': frame}).status_code, 400)

    def test_api_requires_staff_and_csrf(self):
        from django.test import Client
        url = reverse('attendance:capture_frame', args=['STU001'])
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.staff)
        self.assertEqual(client.post(url, {'frame': self.frame_upload()}).status_code, 403)
        self.client.logout()
        self.assertEqual(self.client.post(url, {'frame': self.frame_upload()}).status_code, 302)

    def test_health_is_public_and_does_not_load_models(self):
        self.client.logout()
        with patch('attendance.views.recognize_face_in_frame') as inference:
            self.assertEqual(self.client.get(reverse('attendance:health')).json(), {'status': 'ok'})
            inference.assert_not_called()

    def test_recognition_lock_returns_retry_response(self):
        from attendance.views import inference_lock
        inference_lock.acquire()
        try:
            response = self.client.post(reverse('attendance:capture_frame', args=['STU001']), {'frame': self.frame_upload()})
            self.assertEqual(response.status_code, 429)
        finally:
            inference_lock.release()

    def test_bootstrap_admin_does_not_reset_existing_password(self):
        from django.core.management import call_command
        with patch.dict('os.environ', {'DJANGO_SUPERUSER_USERNAME': 'staff', 'DJANGO_SUPERUSER_PASSWORD': 'different-password-123'}):
            call_command('bootstrap_admin')
        self.staff.refresh_from_db()
        self.assertTrue(self.staff.check_password('test-password'))

    def test_csrf_required_for_training(self):
        from django.test import Client
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.staff)
        self.assertEqual(client.post(reverse('attendance:run_training')).status_code, 403)


class RecognitionTests(SimpleTestCase):
    def test_unanimous_neighbors_do_not_accept_distant_unknown(self):
        encoder = LabelEncoder().fit(['A', 'B'])
        knn = KNeighborsClassifier(n_neighbors=2, weights='distance').fit(
            [[1, 0], [0.99, 0.01], [-1, 0], [-0.99, 0.01]], [0, 0, 1, 1],
        )
        with patch.object(config, 'MAX_EMBEDDING_DISTANCE', 0.8):
            known = classify_embedding([1, 0], knn, encoder)
            unknown = classify_embedding([0, 1], knn, encoder)
        self.assertEqual(known['student_id'], 'A')
        self.assertIsNone(unknown['student_id'])

    def test_capture_rejects_traversal_before_opening_camera(self):
        with patch('face_engine.capture_faces.cv2.VideoCapture') as camera:
            with self.assertRaises(ValueError):
                capture_for_student('../escape')
            camera.assert_not_called()

    def test_detection_clips_negative_coordinates_correctly(self):
        detector = Mock()
        detector.detect_faces.return_value = [{'box': (-10, -10, 30, 30), 'confidence': 0.99}]
        with patch('face_engine.face_utils.FaceNetEngine.detector', return_value=detector):
            result = detect_faces(np.zeros((50, 50, 3), dtype=np.uint8))
        self.assertEqual(result[0]['box'], (0, 0, 20, 20))

    def test_small_training_set_validates_and_saves_atomic_bundle(self):
        labels = ['A'] * 5 + ['B'] * 5
        images = [np.zeros((160, 160, 3), dtype=np.uint8)] * 10
        vectors = np.array([[1, 0]] * 5 + [[0, 1]] * 5, dtype=np.float32)
        with tempfile.TemporaryDirectory(dir=config.BASE_DIR) as directory:
            root = Path(directory)
            with patch.object(config, 'MODEL_DIR', root), patch.object(config, 'MODEL_BUNDLE_PATH', root / 'recognizer.pkl'), patch.object(config, 'TRAINING_REPORT_PATH', root / 'training_report.json'), patch.object(train_model, 'load_dataset', return_value=(images, labels)), patch.object(train_model, 'get_embeddings_batch', return_value=vectors):
                self.assertEqual(train_model.train(), 1.0)
                report = json.loads((root / 'training_report.json').read_text())
                self.assertEqual(report['sample_count'], 10)
                self.assertEqual(report['neighbors'], 4)
                self.assertTrue((root / 'recognizer.pkl').exists())

    def test_training_rejects_insufficient_samples(self):
        with patch.object(train_model, 'load_dataset', return_value=([None] * 4, ['A', 'A', 'B', 'B'])):
            with self.assertRaisesRegex(RuntimeError, 'at least 5'):
                train_model.train()
