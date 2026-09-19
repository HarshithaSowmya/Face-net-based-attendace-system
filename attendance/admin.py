from django.contrib import admin
from .models import Student, Attendance


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ('student_id', 'name', 'department', 'is_enrolled', 'created_at')
    search_fields = ('student_id', 'name', 'department')
    list_filter = ('is_enrolled', 'department')


@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ('student', 'date', 'time_in', 'confidence', 'method')
    list_filter = ('date', 'method')
    search_fields = ('student__name', 'student__student_id')
