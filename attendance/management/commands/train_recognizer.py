from django.core.management.base import BaseCommand, CommandError
from attendance.views import train_registered_students


class Command(BaseCommand):
    help = 'Train FaceNet + KNN on registered students and update enrollment status.'

    def handle(self, *args, **options):
        try:
            accuracy = train_registered_students()
        except Exception as error:
            raise CommandError(str(error)) from error
        self.stdout.write(self.style.SUCCESS(f'Training complete. Validation accuracy: {accuracy:.2%}'))
