#!/bin/sh
cd "$(dirname "$0")/ironpulse" || exit 1
python3 -c "import fastapi, uvicorn, cv2, sklearn, yaml, imageio_ffmpeg" 2>/dev/null   || python3 -m pip install -r requirements.txt || exit 1
[ -f data/ironpulse.db ] || python3 web/baseline_seed.py import
exec python3 run.py
