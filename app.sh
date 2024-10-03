#!/bin/bash
echo "Change direction"
cd /workspaces/X-Ray-Bones-Fracture-Detection/src/api/fracture_detection/
# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate

# Collect static files (if applicable)
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Start the Django development server
echo "Starting the Django development server..."
python manage.py runserver 0.0.0.0:8000
