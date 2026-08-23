@echo off
chcp 65001 >nul
title IronPulse Zero
cd /d "%~dp0ironpulse"

echo.
echo   ============================================
echo     IronPulse Zero - เปิดเว็บแอปตัวจริง
echo   ============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo   [X] ไม่พบ python ในเครื่อง
  echo       ติดตั้ง Python 3.11 จาก https://www.python.org/downloads/ ก่อน
  echo       ตอนติดตั้งให้ติ๊ก "Add python.exe to PATH" ด้วย
  echo.
  pause
  exit /b 1
)

python -c "import fastapi, uvicorn, cv2, sklearn, yaml, imageio_ffmpeg" >nul 2>&1
if errorlevel 1 (
  echo   ยังติดตั้งไลบรารีไม่ครบ กำลังติดตั้งให้ ใช้เวลาสักครู่...
  echo.
  python -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 (
    echo.
    echo   [X] ติดตั้งไลบรารีไม่สำเร็จ
    pause
    exit /b 1
  )
  echo.
)

if not exist "data\ironpulse.db" (
  echo   ยังไม่มีฐานข้อมูล กำลังนำเข้า baseline ที่เตรียมไว้...
  python webaseline_seed.py import
  echo.
)

echo   กำลังเปิดเซิร์ฟเวอร์ เบราว์เซอร์จะเปิดให้เองใน 1-2 วินาที
echo   ปิดหน้าต่างนี้ หรือกด Ctrl+C เพื่อหยุด
echo.
python run.py

echo.
echo   เซิร์ฟเวอร์หยุดทำงานแล้ว
pause
