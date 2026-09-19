"""
attendance/views.py
====================
The Django glue between the web UI and the face_engine package.

Key design choice: heavy face-recognition work (MTCNN + FaceNet + KNN)
lives entirely in `face_engine/`, which knows nothing about Django. These
views only (a) call into face_engine, and (b) read/write the SQLite DB via
the ORM (Student, Attendance). This separation is what lets face_engine
scripts also be run/tested from the plain command line.
"""
import csv
import subprocess
import sys
from datetime import date, timedelta
from io import BytesIO

import cv2
from django.contrib import messages
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.db.models import Count

from .models import Student, Attendance
from face_engine.recognize_attendance import recognize_face_in_frame
from face_engine import train_model as face_trainer
from face_engine import config as face_config


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
def dashboard(request):
    today = timezone.localdate()
    total_students = Student.objects.filter(is_enrolled=True).count()
    present_today = Attendance.objects.filter(date=today).count()
    absent_today = max(total_students - present_today, 0)

    # last 7 days attendance counts, for a small trend chart on the dashboard
    last_7_days = [today - timedelta(days=i) for i in range(6, -1, -1)]
    trend = []
    for d in last_7_days:
        trend.append({
            'date': d.strftime('%b %d'),
            'count': Attendance.objects.filter(date=d).count(),
        })

    context = {
        'total_students': total_students,
        'present_today': present_today,
        'absent_today': absent_today,
        'attendance_rate': round((present_today / total_students * 100), 1) if total_students else 0,
        'trend': trend,
        'recent': Attendance.objects.select_related('student').order_by('-date', '-time_in')[:10],
    }
    return render(request, 'attendance/dashboard.html', context)


# --------------------------------------------------------------------------
# Enrollment: register student -> capture faces -> (re)train model
# --------------------------------------------------------------------------
def register_student(request):
    if request.method == 'POST':
        student_id = request.POST.get('student_id', '').strip()
        name = request.POST.get('name', '').strip()
        department = request.POST.get('department', '').strip()
        email = request.POST.get('email', '').strip()

        if not student_id or not name:
            messages.error(request, "Student ID and Name are required.")
            return redirect('attendance:register')

        student, created = Student.objects.get_or_create(
            student_id=student_id,
            defaults={'name': name, 'department': department, 'email': email},
        )
        if not created:
            messages.warning(request, f"Student {student_id} already exists.")
        else:
            messages.success(request, f"Student {student_id} created. Now capture their face samples.")
        return redirect('attendance:register')

    students = Student.objects.order_by('-created_at')
    return render(request, 'attendance/register.html', {'students': students})


def run_capture(request, student_id):
    """
    Launches face_engine/capture_faces.py as a subprocess. It runs is a
    separate process (not inside the Django worker) because it opens its
    own OpenCV window (cv2.imshow) for the person to see themselves while
    enrolling -- this only works on a machine with a display attached to
    the Django dev server (i.e. running locally, not on a headless server).
    """
    student = get_object_or_404(Student, student_id=student_id)
    try:
        # Run as `python -m face_engine.capture_faces` (module mode) with cwd
        # set to the project root, rather than pointing at the .py file
        # directly -- module mode guarantees the face_engine package is
        # importable, which plain script mode does not.
        subprocess.run(
            [sys.executable, '-m', 'face_engine.capture_faces',
             '--id', student.student_id, '--name', student.name,
             '--samples', str(face_config.SAMPLES_PER_STUDENT)],
            check=True,
            cwd=str(face_config.BASE_DIR),
        )
        messages.success(request, f"Captured face samples for {student.name}. Now click 'Train Model'.")
    except subprocess.CalledProcessError as e:
        messages.error(request, f"Capture failed: {e}")
    return redirect('attendance:register')


def run_training(request):
    """Retrains the KNN classifier on every captured student's images."""
    try:
        accuracy = face_trainer.train()
        Student.objects.update(is_enrolled=True)
        messages.success(request, f"Model retrained successfully. Validation accuracy: {accuracy*100:.1f}%")
    except Exception as e:
        messages.error(request, f"Training failed: {e}")
    return redirect('attendance:register')


# --------------------------------------------------------------------------
# Live, contactless attendance marking via browser (MJPEG stream)
# --------------------------------------------------------------------------
_already_marked_today = set()  # simple in-memory de-dupe guard for this process's lifetime


def gen_frames():
    """Generator yielding MJPEG-encoded frames for the <img> tag in
    mark_attendance.html. For every frame: detect + recognize + draw boxes,
    and write an Attendance row to SQLite the first time each student is
    seen today (enforced twice: in-memory set for speed, and the model's
    unique_together as the real safety net)."""
    cap = cv2.VideoCapture(0)
    today = timezone.localdate()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            for result in recognize_face_in_frame(frame):
                x, y, w, h = result['box']
                student_id = result['student_id']
                confidence = result['confidence']

                if student_id:
                    color = (0, 200, 0)
                    label = f"{student_id} ({confidence:.0%})"
                    key = (student_id, today)
                    if key not in _already_marked_today:
                        student = Student.objects.filter(student_id=student_id).first()
                        if student:
                            _, created = Attendance.objects.get_or_create(
                                student=student, date=today,
                                defaults={'confidence': confidence, 'method': 'face_recognition'},
                            )
                            _already_marked_today.add(key)
                            if created:
                                label += " - MARKED"
                else:
                    color = (0, 0, 220)
                    label = "Unknown"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, max(y - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ok, buffer = cv2.imencode('.jpg', frame)
            if not ok:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()


def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def mark_attendance_page(request):
    return render(request, 'attendance/mark_attendance.html')


# --------------------------------------------------------------------------
# Reports & analytics
# --------------------------------------------------------------------------
def reports(request):
    start = request.GET.get('start') or str(date.today() - timedelta(days=30))
    end = request.GET.get('end') or str(date.today())

    records = (Attendance.objects
               .select_related('student')
               .filter(date__range=[start, end])
               .order_by('-date'))

    per_student = (records.values('student__student_id', 'student__name')
                   .annotate(days_present=Count('id'))
                   .order_by('-days_present'))

    context = {
        'records': records,
        'per_student': per_student,
        'start': start,
        'end': end,
    }
    return render(request, 'attendance/reports.html', context)


def export_csv(request):
    start = request.GET.get('start') or str(date.today() - timedelta(days=30))
    end = request.GET.get('end') or str(date.today())
    records = Attendance.objects.select_related('student').filter(date__range=[start, end]).order_by('date')

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="attendance_{start}_to_{end}.csv"'
    writer = csv.writer(response)
    writer.writerow(['Student ID', 'Name', 'Department', 'Date', 'Time', 'Confidence', 'Method'])
    for r in records:
        writer.writerow([r.student.student_id, r.student.name, r.student.department,
                          r.date, r.time_in, f"{r.confidence:.2f}", r.method])
    return response


def export_pdf(request):
    """Simple tabular PDF report using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    start = request.GET.get('start') or str(date.today() - timedelta(days=30))
    end = request.GET.get('end') or str(date.today())
    records = Attendance.objects.select_related('student').filter(date__range=[start, end]).order_by('date')

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Smart Attendance Report", styles['Title']),
        Paragraph(f"Period: {start} to {end}", styles['Normal']),
        Spacer(1, 12),
    ]

    data = [['Student ID', 'Name', 'Date', 'Time', 'Confidence']]
    for r in records:
        data.append([r.student.student_id, r.student.name, str(r.date),
                     str(r.time_in), f"{r.confidence:.2f}"])

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')]),
    ]))
    elements.append(table)
    doc.build(elements)

    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="attendance_{start}_to_{end}.pdf"'
    return response
