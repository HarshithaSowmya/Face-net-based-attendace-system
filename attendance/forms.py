from django import forms
from django.core.validators import RegexValidator

from .models import Student


class StudentForm(forms.ModelForm):
    student_id = forms.CharField(max_length=30, validators=[RegexValidator(
        r'\A[A-Za-z0-9][A-Za-z0-9_-]*\Z',
        'Use letters, numbers, hyphens and underscores; start with a letter or number.',
    )])

    class Meta:
        model = Student
        fields = ['student_id', 'name', 'department', 'email']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'


class ReportFilterForm(forms.Form):
    start = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    end = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))

    def clean(self):
        cleaned = super().clean()
        if cleaned.get('start') and cleaned.get('end') and cleaned['start'] > cleaned['end']:
            raise forms.ValidationError('Start date must be on or before end date.')
        return cleaned
