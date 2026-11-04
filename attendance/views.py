"""Staff-only enrollment, contactless check-in, analytics and exports."""
import csv
import logging
import re
import uuid
import threading
from datetime import timedelta
from io import BytesIO
from functools import partial
from xml.sax.saxutils import escape

import cv2
from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.db.models import Count
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from django.views.decorators.http import require_POST, require_GET

from .forms import StudentForm, ReportFilterForm
from .models import Student, Attendance
from face_engine.recognize_attendance import recognize_face_in_frame
from face_engine.face_utils import detect_faces, crop_and_align
from .camera import read_uploaded_frame
from face_engine import train_model as face_trainer
from face_engine import config as face_config

staff_member_required = partial(staff_member_required, login_url='login')

inference_lock = threading.Lock()
logger = logging.getLogger(__name__)
training_lock = threading.Lock()


@staff_member_required
def dashboard(request):
    today = timezone.localdate()
    students = Student.objects.filter(is_enrolled=True)
    total = students.count()
    present = Attendance.objects.filter(date=today, student__is_enrolled=True).count()
    days = [today - timedelta(days=i) for i in range(6, -1, -1)]
    counts = dict(Attendance.objects.filter(date__range=[days[0], today])
                  .values('date').annotate(count=Count('id')).values_list('date', 'count'))
    return render(request, 'attendance/dashboard.html', {
        'total_students': total, 'pending_students': Student.objects.filter(is_enrolled=False).count(),
        'present_today': present, 'absent_today': max(total - present, 0),
        'attendance_rate': round(present / total * 100, 1) if total else 0,
        'trend': [{'date': day.strftime('%b %d'), 'count': counts.get(day, 0)} for day in days],
        'trend_max': max(counts.values(), default=1) or 1,
        'recent': Attendance.objects.select_related('student')[:10],
        'model_report': face_trainer.training_report(),
    })


@staff_member_required
def register_student(request):
    form = StudentForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        student = form.save()
        messages.success(request, f'{student.name} registered. Capture face samples next.')
        return redirect('attendance:register')
    students = list(Student.objects.order_by('-created_at'))
    for student in students:
        # Legacy records must also be safe before turning an ID into a filesystem path.
        student.sample_count = (len(list((face_config.DATASET_DIR / student.student_id).glob('*.jpg')))
                                if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', student.student_id) else 0)
    return render(request, 'attendance/register.html', {'students': students, 'form': form})


@staff_member_required
@require_GET
def run_capture(request, student_id):
    student = get_object_or_404(Student, student_id=student_id)
    return render(request, 'attendance/capture.html', {
        'student': student, 'target_samples': face_config.SAMPLES_PER_STUDENT,
    })


@staff_member_required
@require_POST
def capture_frame(request, student_id):
    student = get_object_or_404(Student, student_id=student_id)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,29}', student.student_id):
        return JsonResponse({'error': 'Invalid student identifier.'}, status=400)
    try:
        frame = read_uploaded_frame(request)
    except ValueError as error:
        return JsonResponse({'error': str(error)}, status=400)
    if not inference_lock.acquire(blocking=False):
        return JsonResponse({'error': 'Recognition engine is busy. Please wait.'}, status=429)
    try:
        faces = detect_faces(frame)
        if len(faces) != 1:
            return JsonResponse({'saved': False, 'message': 'Keep exactly one face in view.'})
        directory = face_config.DATASET_DIR / student.student_id
        directory.mkdir(parents=True, exist_ok=True)
        if len(list(directory.glob('*.jpg'))) >= 600:
            return JsonResponse({'error': 'Sample limit reached. Ask an administrator to review existing samples.'}, status=409)
        crop = crop_and_align(frame, faces[0]['box'])
        if crop is None or not cv2.imwrite(str(directory / f'{uuid.uuid4().hex}.jpg'), crop):
            raise RuntimeError('Could not save the sample.')
        Student.objects.filter(pk=student.pk).update(is_enrolled=False)
        return JsonResponse({'saved': True, 'message': 'Sample saved. Slowly vary your pose.'})
    except Exception:
        logger.exception('Browser enrollment failed')
        return JsonResponse({'error': 'Face capture failed. Please ask the administrator to check service logs.'}, status=503)
    finally:
        inference_lock.release()


def train_registered_students():
    if not training_lock.acquire(blocking=False):
        raise RuntimeError('Training is already running.')
    if not inference_lock.acquire(blocking=False):
        training_lock.release()
        raise RuntimeError('Recognition engine is busy. Stop camera sessions and retry.')
    try:
        accuracy = face_trainer.train(set(Student.objects.values_list('student_id', flat=True)))
        report = face_trainer.training_report()
        Student.objects.exclude(student_id__in=report['student_ids']).update(is_enrolled=False)
        Student.objects.filter(student_id__in=report['student_ids']).update(is_enrolled=True)
        return accuracy
    finally:
        inference_lock.release()
        training_lock.release()


@staff_member_required
@require_POST
def run_training(request):
    try:
        accuracy = train_registered_students()
        messages.success(request, f'Model trained. Image-split validation accuracy: {accuracy:.1%}.')
    except Exception:
        logger.exception('Training failed')
        messages.error(request, 'Training failed. Ensure at least two students have five samples each, then check service logs.')
    return redirect('attendance:register')


@staff_member_required
@require_GET
def mark_attendance_page(request):
    return render(request, 'attendance/mark_attendance.html', {
        'model_ready': face_config.MODEL_BUNDLE_PATH.exists(),
    })


@staff_member_required
@require_POST
def recognize_frame(request):
    if not face_config.MODEL_BUNDLE_PATH.exists():
        return JsonResponse({'error': 'Train the recognition model first.'}, status=409)
    try:
        frame = read_uploaded_frame(request)
    except ValueError as error:
        return JsonResponse({'error': str(error)}, status=400)
    if not inference_lock.acquire(blocking=False):
        return JsonResponse({'error': 'Recognition engine is busy. Please wait.'}, status=429)
    try:
        results = []
        for result in recognize_face_in_frame(frame):
            student = Student.objects.filter(student_id=result['student_id'], is_enrolled=True).first() if result['student_id'] else None
            name, marked = 'Unknown', False
            if student:
                now = timezone.localtime()
                _, marked = Attendance.objects.get_or_create(
                    student=student, date=now.date(),
                    defaults={'time_in': now.time(), 'confidence': result['confidence'], 'method': 'face_recognition'},
                )
                name = student.name
            results.append({
                'box': [int(value) for value in result['box']], 'name': name,
                'recognized': student is not None, 'marked': marked,
                'confidence': float(result['confidence']),
            })
        response = JsonResponse({'results': results})
        response['Cache-Control'] = 'no-store'
        return response
    except Exception:
        logger.exception('Browser recognition failed')
        return JsonResponse({'error': 'Recognition failed. Ask the administrator to check service logs.'}, status=503)
    finally:
        inference_lock.release()


@require_GET
def health(request):
    # Lightweight probe: never load TensorFlow or expose student data here.
    return JsonResponse({'status': 'ok'})


def report_range(request):
    today = timezone.localdate()
    form = ReportFilterForm({
        'start': request.GET.get('start') or (today - timedelta(days=30)).isoformat(),
        'end': request.GET.get('end') or today.isoformat(),
    })
    if not form.is_valid():
        return form, None
    return form, Attendance.objects.select_related('student').filter(
        date__range=[form.cleaned_data['start'], form.cleaned_data['end']],
    )


@staff_member_required
def reports(request):
    form, records = report_range(request)
    valid = records is not None
    return render(request, 'attendance/reports.html', {
        'filter_form': form, 'records': records if valid else [],
        'per_student': records.values('student__student_id', 'student__name').annotate(days_present=Count('id')).order_by('-days_present') if valid else [],
        'start': form['start'].value(), 'end': form['end'].value(), 'valid_range': valid,
    })


def csv_cell(value):
    text = str(value)
    return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) else text


@staff_member_required
def export_csv(request):
    form, records = report_range(request)
    if records is None:
        return HttpResponse('Invalid date range.', status=400)
    response = HttpResponse(content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = f'attachment; filename="attendance_{form.cleaned_data["start"]}_to_{form.cleaned_data["end"]}.csv"'
    writer = csv.writer(response)
    writer.writerow(['Student ID', 'Name', 'Department', 'Date', 'Time', 'Confidence', 'Method'])
    for row in records.order_by('date', 'time_in'):
        writer.writerow([csv_cell(row.student.student_id), csv_cell(row.student.name), csv_cell(row.student.department),
                         row.date, row.time_in, f'{row.confidence:.2f}', row.method])
    return response


@staff_member_required
def export_pdf(request):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, LongTable, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    form, records = report_range(request)
    if records is None:
        return HttpResponse('Invalid date range.', status=400)
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=32, rightMargin=32)
    styles = getSampleStyleSheet()
    elements = [Paragraph('Smart Attendance Report', styles['Title']),
                Paragraph(f'Period: {form.cleaned_data["start"]} to {form.cleaned_data["end"]}', styles['Normal']), Spacer(1, 12)]
    data = [['Student ID', 'Name', 'Date', 'Time', 'Confidence']]
    for row in records.order_by('date', 'time_in'):
        data.append([Paragraph(escape(row.student.student_id), styles['BodyText']),
                     Paragraph(escape(row.student.name), styles['BodyText']), str(row.date),
                     row.time_in.strftime('%H:%M:%S'), f'{row.confidence:.2f}'])
    table = LongTable(data, repeatRows=1, colWidths=[90, 160, 85, 65, 65])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#173b36')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eef5f2')]),
    ]))
    elements.append(table)
    doc.build(elements)
    response = HttpResponse(buffer.getvalue(), content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="attendance_report.pdf"'
    return response
