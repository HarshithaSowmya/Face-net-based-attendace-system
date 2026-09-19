from django.db import models
from django.utils import timezone


class Student(models.Model):
    """A person enrolled in the face-recognition system.
    `student_id` doubles as the folder name under media/dataset/<student_id>/
    and as the label the KNN classifier predicts."""
    student_id = models.CharField(max_length=30, unique=True)
    name = models.CharField(max_length=120)
    department = models.CharField(max_length=120, blank=True)
    email = models.EmailField(blank=True)
    is_enrolled = models.BooleanField(
        default=False,
        help_text="True once face samples have been captured AND the model retrained."
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student_id} - {self.name}"


class Attendance(models.Model):
    """One row per (student, date). Marked automatically by the face
    recognition camera feed the first time that student is recognized
    on a given day, so it can't be double-marked."""
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='attendances')
    date = models.DateField(default=timezone.localdate)
    time_in = models.TimeField(default=timezone.now)
    confidence = models.FloatField(help_text="KNN confidence score at time of marking (0-1)")
    method = models.CharField(
        max_length=20,
        default='face_recognition',
        help_text="How attendance was captured, e.g. face_recognition / manual"
    )

    class Meta:
        unique_together = ('student', 'date')   # one attendance record per day
        ordering = ['-date', '-time_in']

    def __str__(self):
        return f"{self.student.name} - {self.date} {self.time_in}"
