"""Create an initial Render administrator without resetting existing passwords."""
import os
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = 'Create the first superuser from DJANGO_SUPERUSER_* variables, if configured.'

    def handle(self, *args, **options):
        username = os.environ.get('DJANGO_SUPERUSER_USERNAME')
        password = os.environ.get('DJANGO_SUPERUSER_PASSWORD')
        if not username or not password:
            self.stdout.write('No bootstrap credentials configured; existing accounts are unchanged.')
            return
        User = get_user_model()
        if User.objects.filter(username=username).exists():
            self.stdout.write('Bootstrap account already exists; password is unchanged.')
            return
        user = User(username=username, email=os.environ.get('DJANGO_SUPERUSER_EMAIL', ''))
        try:
            validate_password(password, user)
        except ValidationError as error:
            raise CommandError('; '.join(error.messages)) from error
        User.objects.create_superuser(username=username, email=user.email, password=password)
        self.stdout.write(self.style.SUCCESS('Initial administrator created.'))
