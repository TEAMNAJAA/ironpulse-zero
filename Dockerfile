FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg

WORKDIR /app

COPY ironpulse/requirements.txt /app/ironpulse/requirements.txt
RUN pip install --no-cache-dir -r /app/ironpulse/requirements.txt

COPY core /app/core
COPY ironpulse/appcore /app/ironpulse/appcore
COPY ironpulse/web /app/ironpulse/web
COPY ironpulse/run.py /app/ironpulse/run.py
COPY ironpulse/config.yaml /app/ironpulse/config.yaml
COPY ironpulse/calibration.json /app/ironpulse/calibration.json
COPY ironpulse/docker-entrypoint.sh /app/ironpulse/docker-entrypoint.sh

RUN chmod +x /app/ironpulse/docker-entrypoint.sh \
    && mkdir -p /app/ironpulse/data/demo /app/ironpulse/data/uploads \
                /app/ironpulse/data/frames \
    && useradd -m -u 10001 ironpulse \
    && chown -R ironpulse:ironpulse /app/ironpulse/data

ENV PYTHONPATH=/app:/app/ironpulse \
    IRONPULSE_PROJECT_ROOT=/app/ironpulse \
    HOST=0.0.0.0 \
    PORT=10000 \
    IRONPULSE_OPEN_BROWSER=false \
    IRONPULSE_MAX_UPLOAD_MB=500 \
    MPLCONFIGDIR=/tmp/mpl

USER ironpulse
WORKDIR /app/ironpulse
EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/api/health'%os.environ.get('PORT','10000'),timeout=4)"

ENTRYPOINT ["/app/ironpulse/docker-entrypoint.sh"]
