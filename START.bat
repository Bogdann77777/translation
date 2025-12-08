@echo off
title Translator - ПУБЛИЧНЫЙ ДОСТУП
color 0A

echo ================================================================================
echo   REAL-TIME ENGLISH-RUSSIAN TRANSLATOR
echo   GPU 0: Local Whisper (STT) + GPU 1: XTTS v2 (TTS)
echo ================================================================================
echo.
echo [1/2] Запускаю сервер на порту 8888...
echo.

REM Запускаем сервер в фоне
start /B "TranslatorServer" "C:\Python311\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8888

echo Ожидание запуска сервера (5 секунд)...
timeout /t 5 /nobreak >nul

echo.
echo [2/2] Создаю публичный туннель (без регистрации)...
echo.
echo ================================================================================
echo   ВАША ПУБЛИЧНАЯ ССЫЛКА ПОЯВИТСЯ НИЖЕ
echo   Ищите строку: https://XXXX.trycloudflare.com
echo ================================================================================
echo.

REM Запускаем Cloudflare туннель
cloudflared tunnel --url http://localhost:8888

pause
