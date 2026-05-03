# ============================================================
# ClassVision-ACAR — Backend Dockerfile
# Python 3.10 + TensorFlow + MediaPipe + YOLO + face_recognition
# ============================================================

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- System Dependencies ----
# cmake & build-essential: required to compile dlib (face_recognition)
# ffmpeg: required for video transcoding
# libgl1, libglib2.0-0: required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Working Directory ----
WORKDIR /app

# ---- Python Dependencies ----
# Copy requirements first for Docker layer caching
COPY requriments.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gunicorn && \
    pip install --no-cache-dir -r requriments.txt

# ---- Application Code ----
COPY . .

# ---- Create required directories ----
RUN mkdir -p media/processed_videos media/csv_uploads media/temp media/logs reports

# ---- Run database migrations ----
RUN python manage.py migrate --noinput 2>/dev/null || true

# ---- Collect static files ----
RUN python manage.py collectstatic --noinput 2>/dev/null || true

# ---- Expose Port ----
EXPOSE 8000

# ---- Start with Gunicorn (production WSGI server) ----
# --timeout 600: video processing can take several minutes
# --workers 2: balance between concurrency and memory usage
CMD ["gunicorn", "classvision.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "600", \
     "--workers", "2", \
     "--threads", "4"]
