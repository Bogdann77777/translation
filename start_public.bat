@echo off
title Translator - PUBLIC ACCESS (No registration needed!)
color 0A

echo ================================================================================
echo   PUBLIC ACCESS - NO REGISTRATION REQUIRED
echo   Using Cloudflare Quick Tunnel (free, no registration)
echo ================================================================================
echo.

echo Starting public tunnel...
echo.
echo Your public URL will appear below (look for "https://xxxx.trycloudflare.com")
echo.
echo Copy and share this URL to access from anywhere!
echo.
echo ================================================================================
echo.

REM Start Cloudflare Quick Tunnel (no registration needed!)
cloudflared tunnel --url http://localhost:8888
