"""
FastAPI приложение для real-time перевода.
"""

import sys
import io

# Force UTF-8 encoding for Windows (fix TTS library charmap error)
# КРИТИЧНО: Должно быть ДО всех импортов!
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import asyncio
import base64
from app.config import load_config
from app.monitoring.logger import setup_logger
from app.monitoring.metrics import MetricsCollector
from app.pipeline.orchestrator import Orchestrator

app = FastAPI(
    title="Real-Time Speech Translator",
    description="English-to-Russian real-time translation",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

logger = setup_logger(__name__)
config = load_config()

# Global metrics collector for API endpoints
global_metrics = MetricsCollector()

# Global orchestrator reference (updated per WebSocket connection)
current_orchestrator = None

# ===========================================
# PRELOADED MODELS (warm up at startup)
# ===========================================
preloaded_whisper = None
preloaded_tts = None
preloaded_llm = None


@app.on_event("startup")
async def startup_preload_models():
    """Preload all ML models at server startup for instant response."""
    global preloaded_whisper, preloaded_tts, preloaded_llm

    logger.info("=" * 60)
    logger.info("PRELOADING MODELS AT STARTUP...")
    logger.info("=" * 60)

    # 1. Load Whisper (STT)
    logger.info("[1/3] Loading Whisper model...")
    try:
        whisper_config = config["models"]["whisper"]
        if whisper_config["provider"] == "local":
            from app.components.local_whisper import LocalWhisperClient
            preloaded_whisper = LocalWhisperClient()
        else:
            from app.components.groq_whisper import GroqWhisperClient
            preloaded_whisper = GroqWhisperClient()
        logger.info("[1/3] Whisper loaded OK")
    except Exception as e:
        logger.error(f"[1/3] Whisper load FAILED: {e}")

    # 2. Load TTS (XTTS or CosyVoice)
    tts_config = config["models"]["tts"]
    tts_provider = tts_config.get("provider", "xtts")
    logger.info(f"[2/3] Loading TTS model ({tts_provider})...")
    try:
        if tts_provider == "cosyvoice":
            from app.components.cosy_voice_engine import CosyVoiceEngine
            preloaded_tts = CosyVoiceEngine()
            logger.info("[2/3] CosyVoice3 loaded OK")
        else:  # default: xtts
            from app.components.xtts_engine import XTTSEngine
            preloaded_tts = XTTSEngine()
            logger.info("[2/3] XTTS loaded OK")
    except Exception as e:
        logger.error(f"[2/3] TTS load FAILED: {e}")

    # 3. Load OpenRouter client (LLM)
    logger.info("[3/3] Loading OpenRouter client...")
    try:
        from app.components.openrouter_llm import OpenRouterClient
        preloaded_llm = OpenRouterClient()
        logger.info("[3/3] OpenRouter loaded OK")
    except Exception as e:
        logger.error(f"[3/3] OpenRouter load FAILED: {e}")

    logger.info("=" * 60)
    logger.info("ALL MODELS PRELOADED - SERVER READY!")
    logger.info("=" * 60)


@app.get("/")
async def root():
    """Главная страница."""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """Health check."""
    return JSONResponse({"status": "healthy"})


@app.get("/metrics")
async def get_metrics():
    """
    Возвращает метрики системы (latency, VRAM, errors).

    Используется для мониторинга и dashboard.
    """
    if current_orchestrator:
        return JSONResponse(current_orchestrator.get_metrics())
    else:
        # Fallback to global metrics if no active session
        summary = global_metrics.get_summary()
        return JSONResponse({
            "status": "no_active_session",
            "global_metrics": summary
        })


@app.get("/status")
async def get_status():
    """
    Возвращает статус очереди батчей (slots).

    Показывает текущее состояние 3-слотовой системы.
    """
    if current_orchestrator and current_orchestrator.batch_queue:
        return JSONResponse(current_orchestrator.batch_queue.get_status())
    else:
        return JSONResponse({
            "status": "no_active_session",
            "slots": []
        })


@app.get("/voices")
async def get_voices():
    """
    Возвращает список доступных голосов из папки voice_samples/.
    """
    import os
    voices_dir = "voice_samples"
    voices = []

    if os.path.exists(voices_dir):
        for f in os.listdir(voices_dir):
            if f.endswith(('.wav', '.mp3')):
                voices.append({
                    "filename": f,
                    "path": f"{voices_dir}/{f}",
                    "name": os.path.splitext(f)[0].replace('_', ' ').replace('-', ' ').title()
                })

    # Sort by name
    voices.sort(key=lambda x: x["name"])

    # Mark current voice
    current_voice = None
    if preloaded_tts:
        current_voice = preloaded_tts.voice_sample

    return JSONResponse({
        "voices": voices,
        "current": current_voice
    })


@app.websocket("/ws/translate")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket для real-time перевода.
    """
    global current_orchestrator

    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")

        # Use preloaded models for instant start
        orchestrator = Orchestrator(
            websocket,
            whisper_client=preloaded_whisper,
            tts_engine=preloaded_tts,
            llm_client=preloaded_llm
        )
        current_orchestrator = orchestrator  # Set global reference for /metrics and /status
        logger.info("Client connected, Orchestrator created with preloaded models")

        message_count = 0
        while True:
            message = await websocket.receive_json()
            message_count += 1
            msg_type = message.get("type")

            logger.debug(f"Received message #{message_count}: type={msg_type}")

            if msg_type == "start":
                mode = message.get("mode", "contextual")
                topic = message.get("topic", None)
                logger.info(f"Processing 'start' message (mode: {mode}, topic: {topic or 'none'})")
                await orchestrator.start_session(mode=mode, topic=topic)

            elif msg_type == "audio":
                audio_data_b64 = message.get("data", "")
                audio_bytes = base64.b64decode(audio_data_b64)
                logger.debug(f"Processing 'audio' message: {len(audio_bytes)} bytes")
                await orchestrator.process_audio(audio_bytes)

            elif msg_type == "stop":
                logger.info("Processing 'stop' message")
                await orchestrator.stop_session()

            elif msg_type == "set_speed":
                new_speed = message.get("speed", 1.0)
                logger.info(f"Setting TTS speed to {new_speed}x")
                if preloaded_tts:
                    preloaded_tts.speed = new_speed

            elif msg_type == "set_voice":
                new_voice = message.get("voice", "")
                logger.info(f"Switching voice to: {new_voice}")
                if preloaded_tts and new_voice:
                    preloaded_tts.voice_sample = new_voice
                    await websocket.send_json({
                        "type": "voice_changed",
                        "voice": new_voice
                    })

            else:
                logger.warning(f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"Client disconnected (processed {message_count} messages)")
        if orchestrator.session_active:
            logger.info("Stopping active session on disconnect")
            await orchestrator.stop_session()
        current_orchestrator = None  # Clear global reference

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)

        # Детальное логирование ошибки
        logger.error(f"WebSocket error: {error_type}: {error_msg}", exc_info=True)

        # Проверяем специфичные ошибки Windows 11 / CUDA
        if "cuda" in error_msg.lower() or "out of memory" in error_msg.lower():
            logger.critical(f"CUDA/VRAM error detected! This may be Windows 11 specific. Error: {error_msg}")
            logger.critical("SOLUTION: Try reducing batch_queue_size in config.yaml or use smaller model")

            # Очищаем CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared after error")
            except Exception as cuda_err:
                logger.error(f"Failed to clear CUDA cache: {cuda_err}")

        # Останавливаем сессию если активна
        try:
            if orchestrator.session_active:
                logger.info("Stopping active session after error")
                await orchestrator.stop_session()
        except Exception as stop_err:
            logger.error(f"Error stopping session: {stop_err}")

        current_orchestrator = None

        # НЕ пробрасываем ошибку дальше - graceful shutdown
        # raise


if __name__ == "__main__":
    server_config = config["server"]
    uvicorn.run(
        app,
        host=server_config["host"],
        port=server_config["port"],
        log_level="info"
    )
