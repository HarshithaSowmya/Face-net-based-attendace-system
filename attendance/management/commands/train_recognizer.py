"""
Django management command wrapper so training can also be run as:
    python manage.py train_recognizer
(equivalent to calling face_engine/train_model.py directly, but integrated
with `manage.py` like any other Django admin task, and it marks matching
Students as enrolled in one step.)
"""
from django.core.management.base import BaseCommand
from attendance.models import Student
from face_engine import train_model


class Command(BaseCommand):
    help = "Extract FaceNet embeddings for all captured face images and (re)train the KNN classifier."

    def handle(self, *args, **options):
        accuracy = train_model.train()
        Student.objects.update(is_enrolled=True)
        self.stdout.write(self.style.SUCCESS(f"Training complete. Validation accuracy: {accuracy*100:.2f}%"))
