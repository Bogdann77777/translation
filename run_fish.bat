@echo off
setlocal enabledelayedexpansion
title Translator [FishSpeech 1.5] - Starting...
color 0B

echo.
echo ================================================================================
echo   REAL-TIME TRANSLATOR  [Fish Speech 1.5]
echo ================================================================================
echo.

REM Install ormsgpack if not present
echo   Checking ormsgpack...
"E:\crewai\translator\venv\Scripts\python.exe" -c "import ormsgpack" 2>nul
if errorlevel 1 (
    echo   Installing ormsgpack...
    "E:\crewai\translator\venv\Scripts\pip.exe" install ormsgpack -q
    echo   ormsgpack installed.
) else (
    echo   ormsgpack OK
)

REM Kill old fish-speech server on port 8080
echo   Stopping old fish-speech server (if any)...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8080 "') do (
    if not "%%a"=="0" taskkill /F /PID %%a >nul 2>&1
)

REM Start fish-speech 1.5 server in a separate window
echo   Starting fish-speech 1.5 server...
start "FishSpeech-1.5" cmd /k "cd /d E:\crewai\translator\fish-speech && set PYTHONIOENCODING=utf-8 && E:\crewai\translator\venv-fish\Scripts\python tools\api_server.py --llama-checkpoint-path checkpoints/fish-speech-1.5 --decoder-checkpoint-path checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth --decoder-config-name firefly_gan_vq --device cuda --listen 0.0.0.0:8080"

REM Wait for model to load (~20-30 seconds)
echo   Waiting 30s for fish-speech to load...
timeout /t 30 /nobreak >nul

REM Check readiness (retry up to 6 times, 5s each)
set /a RETRIES=0
:wait_loop
curl -sf http://localhost:8080/ >nul 2>&1
if not errorlevel 1 goto fish_ready
set /a RETRIES+=1
if %RETRIES% GEQ 6 (
    echo   WARNING: fish-speech may not be ready yet, continuing anyway...
    goto fish_ready
)
echo   Still waiting for fish-speech... (%RETRIES%/6)
timeout /t 5 /nobreak >nul
goto wait_loop

:fish_ready
echo   fish-speech server is READY!

REM Set TTS provider to fish_speech
set TRANSLATOR_TTS_PROVIDER=fish_speech
set PYTHONIOENCODING=utf-8

REM -----------------------------------------------------------------------
REM Everything below is identical to RUN.bat
REM -----------------------------------------------------------------------

REM Kill any existing ngrok processes
taskkill /F /IM ngrok.exe >nul 2>&1

REM Get local IP
set LOCAL_IP=
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do (
        if not defined LOCAL_IP set LOCAL_IP=%%b
    )
)

REM Start ngrok in background
echo   Starting ngrok tunnel...
start "" ngrok http 8888

REM Wait for ngrok to initialize
timeout /t 3 /nobreak >nul

REM Get public URL from ngrok API
set PUBLIC_URL=
for /f "usebackq delims=" %%a in (`"C:\Python311\python.exe" -c "import urllib.request,json; r=urllib.request.urlopen('http://127.0.0.1:4040/api/tunnels'); d=json.loads(r.read()); print([t['public_url'] for t in d['tunnels'] if 'https' in t['public_url']][0] if d['tunnels'] else 'Not available')" 2^>nul`) do set PUBLIC_URL=%%a

if "!PUBLIC_URL!"=="" set PUBLIC_URL=Check ngrok window

echo.
echo   -----------------------------------------------
echo   LOCAL:    http://localhost:8888
echo   NETWORK:  http://%LOCAL_IP%:8888
echo   PUBLIC:   !PUBLIC_URL!
echo   TTS:      Fish Speech 1.5 (port 8080)
echo   -----------------------------------------------
echo.
echo   Loading models... (~30 sec)
echo.

REM Open browser
start "" http://localhost:8888

REM Change title
title Translator [FishSpeech 1.5] - Running

REM Start server
"C:\Python311\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8888

REM Cleanup
echo.
echo Shutting down...
taskkill /F /IM ngrok.exe >nul 2>&1
echo Done.
