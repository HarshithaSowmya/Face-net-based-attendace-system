#!/usr/bin/env bash
set -o errexit
mkdir -p "$DATA_DIR"
# Render mounts persistent disks only at runtime, not during build/pre-deploy.
python manage.py migrate --noinput
python manage.py bootstrap_admin
exec gunicorn attendance_system.wsgi:application --bind "0.0.0.0:${PORT:-10000}" --workers 1 --threads 4 --timeout 600 --access-logfile - --error-logfile -
