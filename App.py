import os
import subprocess
import sys

def activate_virtualenv():
    """Activate the virtual environment."""
    venv_activate = os.path.join('env', 'bin', 'activate_this.py')
    if os.path.exists(venv_activate):
        with open(venv_activate) as file_:
            exec(file_.read(), dict(__file__=venv_activate))
        print("Virtual environment activated.")
    else:
        print("Virtual environment not found.")
        sys.exit(1)

def run_command(command):
    """Run a shell command."""
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {command}")
        sys.exit(1)

def apply_migrations():
    """Apply database migrations."""
    print("Applying database migrations...")
    run_command("python manage.py migrate")

def collect_static_files():
    """Collect static files."""
    print("Collecting static files...")
    run_command("python manage.py collectstatic --noinput")

def start_server():
    """Start the Django development server."""
    print("Starting Django server...")
    run_command("python manage.py runserver 0.0.0.0:8000")

if __name__ == "__main__":
    activate_virtualenv()
    apply_migrations()
    collect_static_files()
    start_server()
