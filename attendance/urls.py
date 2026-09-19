from django.urls import path
from . import views

app_name = 'attendance'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('register/', views.register_student, name='register'),
    path('capture/<str:student_id>/', views.run_capture, name='run_capture'),
    path('train/', views.run_training, name='run_training'),
    path('mark/', views.mark_attendance_page, name='mark_attendance'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('reports/', views.reports, name='reports'),
    path('reports/export/csv/', views.export_csv, name='export_csv'),
    path('reports/export/pdf/', views.export_pdf, name='export_pdf'),
]
