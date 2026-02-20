"""
Модуль главного координатора системы.
"""

import asyncio
import time
from typing import Dict, Any
from app.config import load_config
from app.monitoring.logger import setup_logger, log_json
from app.monitoring.metrics import MetricsCollector
from app.pipeline.batch_queue import BatchQueue
from app.pipeline.stream_processor import StreamProcessor
from app.pipeline.literal_stream_processor import LiteralStreamProcessor


class Orchestrator:
    """
    Главный координатор всей системы перевода.
    """

    def __init__(self, websocket, whisper_client=None, tts_engine=None, llm_client=None):
        """Инициализация оркестратора."""
        self.config = load_config()["pipeline"]
        self.logger = setup_logger(__name__)
        self.metrics = MetricsCollector()
        self.websocket = websocket

        # Store preloaded models
        self.whisper_client = whisper_client
        self.tts_engine = tts_engine
        self.llm_client = llm_client

        self.batch_queue = None
        self.stream_processor = None
        self.session_active = False
        self.restart_count = 0
        self.translation_mode = 'contextual'  # 'contextual' or 'literal'

        self.logger.info("Orchestrator initialized")

    async def start_session(self, mode: str = 'contextual') -> None:
        """
        Запускает новую сессию.

        Args:
            mode: Режим перевода ('contextual' или 'literal')
        """
        self.translation_mode = mode
        log_json(self.logger, "INFO", "Starting session",
                 session_id=id(self), mode=mode, timestamp=time.time())

        # Pass preloaded models to BatchQueue
        self.batch_queue = BatchQueue(
            self.websocket,
            whisper_client=self.whisper_client,
            tts_engine=self.tts_engine,
            llm_client=self.llm_client,
            metrics_collector=self.metrics
        )

        # Выбираем процессор в зависимости от режима
        if mode == 'literal':
            self.stream_processor = LiteralStreamProcessor(self.batch_queue)
            self.logger.info("🚀 LITERAL mode: Fast processing (2-5s chunks, 0.2s pauses)")
        else:
            self.stream_processor = StreamProcessor(self.batch_queue)
            self.logger.info("🎯 CONTEXTUAL mode: Quality processing (12-18s chunks, 1.0s pauses)")

        self.session_active = True

        # Запускаем фоновый цикл воспроизведения
        await self.batch_queue.start_playback_loop()

        await self.websocket.send_json({
            "type": "session_started",
            "mode": mode,
            "timestamp": time.time()
        })
    
    async def stop_session(self) -> None:
        """Останавливает сессию."""
        metrics_summary = self.metrics.get_summary()
        log_json(self.logger, "INFO", "Stopping session",
                 session_id=id(self),
                 batches_processed=self.metrics.batches_processed,
                 session_duration=metrics_summary["session_duration"],
                 errors=metrics_summary["errors"])

        self.session_active = False

        if self.batch_queue:
            # Останавливаем фоновый цикл воспроизведения
            await self.batch_queue.stop_playback_loop()
            await self.batch_queue.context_buffer.clear()

        self.batch_queue = None
        self.stream_processor = None
    
    async def process_audio(self, audio_bytes: bytes) -> None:
        """
        Обрабатывает входящий аудио чанк.
        """
        if not self.session_active:
            return
        
        try:
            await self.stream_processor.process_chunk(audio_bytes)
        except Exception as e:
            log_json(self.logger, "ERROR", "Audio processing error",
                     error_type=type(e).__name__,
                     error_message=str(e),
                     chunk_size=len(audio_bytes))
            self.metrics.record_error("audio_processing", str(e))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Возвращает метрики системы."""
        summary = self.metrics.get_summary()
        return {
            "session_active": self.session_active,
            "batches_processed": self.metrics.batches_processed,
            "uptime": time.time() - self.metrics.session_start,
            "latency": summary["latency_avg"],
            "errors": summary["errors"],
            "vram_mb": summary["vram_current_mb"]
        }
