"""
Модуль управления очередью батчей (3 слота).
3-слотовая система для non-stop обработки.
"""

import asyncio
import time
import base64
import numpy as np
from typing import Dict, Any, Optional
from app.config import load_config
from app.monitoring.logger import setup_logger, log_json
from app.monitoring.metrics import MetricsCollector
from app.pipeline.context_buffer import ContextBuffer
from app.components.groq_whisper import GroqWhisperClient
from app.components.local_whisper import LocalWhisperClient
from app.components.openrouter_llm import OpenRouterClient
from app.components.xtts_engine import XTTSEngine


class BatchQueue:
    """
    3-слотовая очередь батчей для non-stop обработки.
    
    Концепция:
        Slot 1: PLAYING - Воспроизведение аудио
        Slot 2: READY - Готов к воспроизведению  
        Slot 3: PROCESSING - Обработка (STT → LLM → TTS)
    
    Принцип:
        Пока Slot 1 играет, Slot 3 обрабатывает новый батч.
        Когда Slot 1 освободился → Slot 2 → Slot 1 (playing).
    """
    
    def __init__(self, websocket, whisper_client=None, tts_engine=None, llm_client=None):
        """
        Инициализация очереди батчей.

        Args:
            websocket: WebSocket connection
            whisper_client: Preloaded Whisper client (optional)
            tts_engine: Preloaded TTS engine (optional)
            llm_client: Preloaded LLM client (optional)
        """
        self.config = load_config()["pipeline"]
        self.logger = setup_logger(__name__)
        self.metrics = MetricsCollector()
        self.websocket = websocket

        # Use preloaded models if provided, otherwise create new
        if whisper_client:
            self.whisper_client = whisper_client
            self.logger.info("Using preloaded Whisper client")
        else:
            whisper_config = load_config()["models"]["whisper"]
            if whisper_config["provider"] == "local":
                self.whisper_client = LocalWhisperClient()
            else:
                self.whisper_client = GroqWhisperClient()

        if llm_client:
            self.openrouter_client = llm_client
            self.logger.info("Using preloaded LLM client")
        else:
            self.openrouter_client = OpenRouterClient()

        if tts_engine:
            self.xtts_engine = tts_engine
            self.logger.info("Using preloaded TTS engine")
        else:
            self.xtts_engine = XTTSEngine()

        self.context_buffer = ContextBuffer()

        # НОВАЯ АРХИТЕКТУРА: Очередь готовых батчей (FIFO)
        self.ready_queue = asyncio.Queue()  # Неограниченная очередь готовых батчей
        self.playback_task = None  # Фоновая задача воспроизведения
        self.is_running = False

        # Счетчики для мониторинга
        self.processing_count = 0  # Сколько батчей сейчас обрабатывается
        self.processing_lock = asyncio.Lock()  # Для атомарности счетчика

        self.logger.info("BatchQueue initialized (parallel queue system)")
    
    async def add_batch(self, audio_array: np.ndarray) -> None:
        """
        Добавляет новый батч аудио в очередь на обработку.

        ПОЛНОСТЬЮ НЕ блокирующая операция - НЕМЕДЛЕННО запускает обработку в фоне.
        Несколько батчей могут обрабатываться параллельно.

        Args:
            audio_array: Numpy массив с аудио (float32, 16kHz)
        """
        # Увеличиваем счетчик обрабатываемых батчей
        async with self.processing_lock:
            self.processing_count += 1

        # Запускаем обработку В ФОНЕ (асинхронно, БЕЗ ОЖИДАНИЯ)
        asyncio.create_task(self._process_batch_async(audio_array))

        self.logger.debug(f"Batch queued for processing (total processing: {self.processing_count})")

    async def _process_batch_async(self, audio_array: np.ndarray) -> None:
        """
        Фоновая обработка батча через полный pipeline.

        Обрабатывает батч (STT → LLM → TTS) и кладет результат в ready_queue.
        Выполняется полностью асинхронно, не блокируя другие батчи.

        Args:
            audio_array: Numpy массив с аудио (float32, 16kHz)
        """
        try:
            # Обрабатываем батч через полный pipeline (5-10 секунд)
            processed = await self.process_batch(audio_array)

            # Кладем готовый батч в очередь воспроизведения
            await self.ready_queue.put(processed)

            self.logger.debug("Batch processed and queued for playback")

        except Exception as e:
            self.logger.error(f"Background batch processing failed: {e}")
            self.metrics.record_error("batch_processing_async", str(e))

        finally:
            # Уменьшаем счетчик обрабатываемых батчей
            async with self.processing_lock:
                self.processing_count -= 1

    async def start_playback_loop(self) -> None:
        """
        Запускает фоновый цикл воспроизведения.

        Этот цикл постоянно берет готовые батчи из ready_queue
        и воспроизводит их последовательно (non-stop).

        Должен быть вызван один раз при старте сессии.
        """
        if self.is_running:
            self.logger.warning("Playback loop already running")
            return

        self.is_running = True
        self.playback_task = asyncio.create_task(self._playback_loop())
        self.logger.info("Playback loop started")

    async def _playback_loop(self) -> None:
        """
        Внутренний цикл воспроизведения.

        Постоянно берет батчи из ready_queue и воспроизводит.
        Работает пока self.is_running == True.
        """
        self.logger.info("Playback loop running")

        while self.is_running:
            try:
                # Берем следующий готовый батч из очереди (ждем если пусто)
                batch = await self.ready_queue.get()

                # Воспроизводим
                await self._play_batch(batch)

                # Помечаем задачу как выполненную
                self.ready_queue.task_done()

            except asyncio.CancelledError:
                self.logger.info("Playback loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Playback loop error: {e}")

        self.logger.info("Playback loop stopped")

    async def _play_batch(self, batch: Dict[str, Any]) -> None:
        """
        Воспроизводит один батч (отправка клиенту через WebSocket).

        Args:
            batch: Обработанный батч с полями:
                - original: английский текст
                - translated: русский перевод
                - audio: WAV байты
                - duration: длительность аудио
        """
        batch_num = self.metrics.batches_processed + 1
        queue_size = self.ready_queue.qsize()

        self.logger.info(f"=== PLAYING BATCH #{batch_num} (duration: {batch['duration']:.1f}s, queue: {queue_size} waiting) ===")

        # Отправляем транскрипцию
        await self.websocket.send_json({
            "type": "transcription",
            "text": batch["original"],
            "timestamp": time.time()
        })

        # Отправляем перевод
        await self.websocket.send_json({
            "type": "translation",
            "original": batch["original"],
            "translated": batch["translated"],
            "timestamp": time.time()
        })

        # Отправляем аудио
        await self.websocket.send_json({
            "type": "audio_output",
            "data": base64.b64encode(batch["audio"]).decode(),
            "duration": batch["duration"],
            "timestamp": time.time()
        })

        # Ждём окончания воспроизведения
        await asyncio.sleep(batch["duration"])

        # Увеличиваем счётчик обработанных батчей
        self.metrics.batches_processed += 1

        self.logger.info(f"=== BATCH #{batch_num} DONE (played {batch['duration']:.1f}s) ===")

    async def stop_playback_loop(self) -> None:
        """
        Останавливает фоновый цикл воспроизведения.

        Должен быть вызван при остановке сессии.
        """
        self.is_running = False

        if self.playback_task:
            self.playback_task.cancel()
            try:
                await self.playback_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Playback loop stopped")

    async def process_batch(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """
        Обрабатывает батч через STT → LLM → TTS pipeline.
        
        КРИТИЧНО: При любой ошибке → raise Exception → RESTART!
        
        Args:
            audio_array: Аудио массив (float32, 16kHz)
        
        Returns:
            Dict с результатами обработки:
                {
                    "original": "английский текст",
                    "translated": "русский перевод",
                    "audio": bytes,  # WAV аудио
                    "duration": float,  # длительность аудио
                    "timestamp": float
                }
        """
        try:
            # STEP 1: STT (Local Whisper on GPU or Groq)
            start = time.time()
            transcription = await self.whisper_client.transcribe(audio_array)
            stt_duration = time.time() - start
            self.metrics.record_latency("stt", stt_duration)

            # NOTE: Transcription will be sent to client only when batch starts playing (Slot 1)
            
            # STEP 2: Translation (OpenRouter + context)
            start = time.time()
            context = await self.context_buffer.get_context()
            translation = await self.openrouter_client.translate(
                transcription["text"], context
            )
            translation_duration = time.time() - start
            self.metrics.record_latency("translation", translation_duration)

            # NOTE: Translation will be sent to client only when batch starts playing (Slot 1)
            
            # Добавляем в контекст (для следующих переводов)
            await self.context_buffer.add_sentence(transcription["text"])
            
            # STEP 3: TTS (XTTS-v2)
            start = time.time()
            audio_bytes = await self.xtts_engine.synthesize(translation)
            tts_duration = time.time() - start
            self.metrics.record_latency("tts", tts_duration)

            # Вычисляем длительность аудио из WAV header
            # WAV format: 44 bytes header + data
            # Sample rate берём из конфига TTS
            tts_sample_rate = self.xtts_engine.output_sample_rate
            audio_data_size = len(audio_bytes) - 44  # Subtract WAV header
            audio_duration = audio_data_size / (tts_sample_rate * 2)  # 2 bytes/sample (int16)
            
            # E2E метрика
            e2e_duration = stt_duration + translation_duration + tts_duration
            self.metrics.record_latency("e2e", e2e_duration)
            
            # Логируем успешную обработку
            log_json(self.logger, "INFO", "Batch processed",
                     stt=stt_duration, translation=translation_duration,
                     tts=tts_duration, e2e=e2e_duration)
            
            # Возвращаем результат
            return {
                "original": transcription["text"],
                "translated": translation,
                "audio": audio_bytes,
                "duration": audio_duration,
                "timestamp": time.time()
            }
        
        except Exception as e:
            # Записываем ошибку
            self.metrics.record_error("batch_processing", str(e))
            self.logger.error(f"Batch processing failed: {e}")
            raise  # Пробрасываем выше → RESTART!

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус очереди (для UI dashboard).

        Returns:
            Dict с информацией о состоянии обработки:
                {
                    "processing_count": 2,  # Сколько батчей обрабатывается
                    "ready_queue_size": 3,  # Сколько готовых батчей ждут воспроизведения
                    "playback_active": True,  # Работает ли цикл воспроизведения
                    "slots": [...]  # Для совместимости с UI
                }

        Note:
            Метод синхронный для совместимости с HTTP endpoint.
        """
        # Формируем псевдо-слоты для обратной совместимости с UI
        slots_status = []

        # Эмулируем слоты на основе реального состояния
        if self.is_running:
            slots_status.append({"slot": 1, "status": "playing"})

        ready_count = self.ready_queue.qsize()
        if ready_count > 0:
            slots_status.append({"slot": 2, "status": "ready"})

        if self.processing_count > 0:
            slots_status.append({"slot": 3, "status": "processing"})

        return {
            "processing_count": self.processing_count,
            "ready_queue_size": ready_count,
            "playback_active": self.is_running,
            "slots": slots_status
        }
