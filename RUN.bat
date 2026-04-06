@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion
title Real-Time Translator - Starting...
color 0A

echo.
echo ================================================================================
echo   REAL-TIME ENGLISH-RUSSIAN TRANSLATOR
echo ================================================================================
echo.

REM Kill any existing cloudflared processes
taskkill /F /IM cloudflared.exe >nul 2>&1

REM Get local IP
set LOCAL_IP=
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do (
        if not defined LOCAL_IP set LOCAL_IP=%%b
    )
)

REM Start cloudflared tunnel in background, log to file
echo   Starting cloudflared tunnel...
if not exist "%~dp0logs" mkdir "%~dp0logs"
start "" /B cmd /c "cloudflared tunnel --url localhost:8888 --no-autoupdate >> ""%~dp0logs\cloudflared.log"" 2>&1"

REM Wait for cloudflared to get URL (~8 sec)
echo   Waiting for tunnel URL...
timeout /t 9 /nobreak >nul

REM Extract public URL from cloudflared log
set PUBLIC_URL=
for /f "usebackq delims=" %%a in (`"C:\Python311\python.exe" -c "import re,sys; log=open(r'%~dp0logs\cloudflared.log').read(); m=re.search(r'https://[a-z0-9\-]+\.trycloudflare\.com', log); print(m.group(0) if m else '')" 2^>nul`) do set PUBLIC_URL=%%a

if "!PUBLIC_URL!"=="" set PUBLIC_URL=Check logs\cloudflared.log

echo.
echo   -----------------------------------------------
echo   LOCAL:    http://localhost:8888
echo   NETWORK:  http://%LOCAL_IP%:8888
echo   PUBLIC:   !PUBLIC_URL!
echo   -----------------------------------------------
echo.
echo   Share PUBLIC URL to access from anywhere!
echo.
echo ================================================================================
echo.
echo   Loading models... (~2 min)
echo.

REM Open browser
start "" http://localhost:8888

REM Change title
title Real-Time Translator - Running  [!PUBLIC_URL!]

REM Start server
set PYTHONIOENCODING=utf-8
"C:\Python311\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8888

REM Cleanup
echo.
echo Shutting down...
taskkill /F /IM cloudflared.exe >nul 2>&1
echo Done.
pause
