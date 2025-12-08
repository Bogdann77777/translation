@echo off
setlocal enabledelayedexpansion
title Real-Time Translator - Starting...
color 0A

echo.
echo ================================================================================
echo   REAL-TIME ENGLISH-RUSSIAN TRANSLATOR
echo ================================================================================
echo.

REM Kill any existing ngrok processes
taskkill /F /IM ngrok.exe >nul 2>&1

REM Get local IP
set LOCAL_IP=
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do (
        if not defined LOCAL_IP set LOCAL_IP=%%b
    )
)

REM Start ngrok in background and save output
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
echo   -----------------------------------------------
echo.
echo   Share PUBLIC URL to access from anywhere!
echo   (ngrok window is open separately)
echo.
echo ================================================================================
echo.
echo   Loading models... (~2 min)
echo.

REM Open browser
start "" http://localhost:8888

REM Change title
title Real-Time Translator - Running

REM Start server
"C:\Python311\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8888

REM Cleanup
echo.
echo Shutting down...
taskkill /F /IM ngrok.exe >nul 2>&1
echo Done.
