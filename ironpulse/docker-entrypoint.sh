#!/bin/sh
set -e

echo "IronPulse Zero"
echo "  ffmpeg    : $(ffmpeg -version 2>/dev/null | head -1)"
echo "  python    : $(python -V 2>&1)"
echo "  data dir  : ${IRONPULSE_DATA_DIR:-data}"
echo "  port      : ${PORT:-10000}"

python web/baseline_seed.py import || echo "  (ข้ามการนำเข้า baseline)"

exec uvicorn web.app:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-10000}" \
    --workers 1 \
    --timeout-keep-alive 120 \
    --log-level warning
