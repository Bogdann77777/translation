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
from app.components.tts_worker_pool import TTSWorkerPool


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
    
    def __init__(self, websocket, whisper_client=None, tts_engine=None, llm_client=None, metrics_collector=None, topic=None):
        """
        Инициализация очереди батчей.

        Args:
            websocket: WebSocket connection
            whisper_client: Preloaded Whisper client (optional)
            tts_engine: Preloaded TTS engine (optional)
            llm_client: Preloaded LLM client (optional)
            metrics_collector: Shared metrics collector (optional)
            topic: Optional topic/context for translation (optional)
        """
        self.config = load_config()["pipeline"]
        self.logger = setup_logger(__name__)
        self.metrics = metrics_collector if metrics_collector else MetricsCollector()
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

        # TTS Worker Pool: 2 workers на 2 GPU для параллельной обработки
        # (решает проблему CUDA assertion errors при concurrent requests)
        self._tts_is_preloaded = False  # Track if pool is shared (don't shutdown)

        if tts_engine:
            # Check if it's a TTSWorkerPool instance (preloaded at startup)
            if isinstance(tts_engine, TTSWorkerPool):
                self.tts_worker_pool = tts_engine
                self.xtts_engine = None
                self._tts_is_preloaded = True  # PRELOADED - don't shutdown!
                self.logger.info("Using PRELOADED TTS Worker Pool (instant startup!)")
            else:
                # Legacy mode: direct XTTSEngine
                self.xtts_engine = tts_engine
                self.tts_worker_pool = None
                self.logger.info("Using preloaded TTS engine (legacy mode)")
        else:
            # Создаём worker pool с 2 workers на 2 GPU
            self.tts_worker_pool = TTSWorkerPool(num_workers=2)
            self.xtts_engine = None  # Не используем direct engine
            self.logger.info("Using TTS Worker Pool (2 workers on GPU 0 and GPU 1) - 20s startup delay")

        self.context_buffer = ContextBuffer()

        # Topic/context for better translation accuracy
        self.topic = topic

        # НОВАЯ АРХИТЕКТУРА: Очередь готовых батчей (FIFO)
        self.max_ready_queue_size = self.config.get("max_ready_queue_size", 10)  # Max chunks in queue
        self.ready_queue = asyncio.Queue()  # Очередь готовых батчей
        self.playback_task = None  # Фоновая задача воспроизведения
        self.is_running = False

        # SEQUENTIAL PLAYBACK: Buffer for out-of-order chunks
        self.completed_chunks_buffer = {}  # {chunk_id: batch_data}
        self.next_playback_chunk_id = 1  # Next chunk ID to play (sequential order)

        # NON-STOP PLAYBACK: Минимум готовых чанков перед стартом воспроизведения
        # Это критично для дословного режима (маленькие чанки)
        self.min_ready_chunks_before_start = self.config.get("min_ready_chunks_before_start", 3)
        self.playback_started = False  # Флаг первого запуска

        # ADAPTIVE SPEED: Auto-adjust TTS speed based on queue size
        tts_config = load_config()["models"]["tts"]
        self.adaptive_speed_config = tts_config.get("adaptive_speed", {})
        self.adaptive_speed_enabled = self.adaptive_speed_config.get("enabled", False)
        if self.adaptive_speed_enabled:
            self.logger.info(
                f"Adaptive speed ENABLED: {self.adaptive_speed_config['min_speed']}x - "
                f"{self.adaptive_speed_config['max_speed']}x based on queue size"
            )

        # Счетчики для мониторинга
        self.processing_count = 0  # Сколько батчей сейчас обрабатывается
        self.processing_lock = asyncio.Lock()  # Для атомарности счетчика
        self.chunk_counter = 0  # Глобальный счётчик чанков для отслеживания (CHUNK #1, #2, #3...)

        # PIPELINE CONCURRENCY CONTROL
        # WHISPER: semaphore=1 (sequential) - preserve chunk order
        # TRANSLATION: semaphore=2-3 (parallel) - API, no CUDA, can handle multiple requests
        # TTS: NO SEMAPHORE! Worker pool handles parallelism via queue + separate processes
        self.whisper_semaphore = asyncio.Semaphore(1)  # Sequential (preserve order)
        self.translation_semaphore = asyncio.Semaphore(3)  # Parallel (API) - increased to 3
        # self.tts_semaphore removed - worker pool handles this now!

        # GLOBAL PIPELINE LIMIT: Максимум N батчей в системе одновременно
        # (Processing + Ready + Playing)
        self.max_concurrent_batches = self.config.get("batch_queue_size", 3)
        self.pipeline_semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        self.logger.info(
            f"BatchQueue initialized (pipeline: max {self.max_concurrent_batches} batches, "
            f"NON-STOP: buffer {self.min_ready_chunks_before_start} chunks before playback, "
            f"STT→Translation→TTS→Playback)"
        )
    
    async def add_batch(self, audio_array: np.ndarray) -> None:
        """
        Добавляет новый батч аудио в очередь на обработку.

        ОЧЕРЕДЬ С ОГРАНИЧЕНИЕМ: Ждет свободный слот если уже обрабатывается
        максимальное количество батчей (batch_queue_size из config).

        Это гарантирует, что в системе будет не больше N батчей одновременно
        (обработка + готовые + воспроизведение).

        Args:
            audio_array: Numpy массив с аудио (float32, 16kHz)
        """
        # КРИТИЧНО: Ждем свободный слот в pipeline (блокирующая операция!)
        # Это ограничивает общее количество батчей в системе

        # Check if we need to wait (все слоты заняты)
        if self.pipeline_semaphore._value == 0:
            self.logger.warning(f"⚠️ Pipeline FULL ({self.max_concurrent_batches}/{self.max_concurrent_batches} slots) - waiting for free slot... (processing: {self.processing_count}, ready: {self.ready_queue.qsize()})")

        await self.pipeline_semaphore.acquire()

        # Присваиваем уникальный ID чанку
        async with self.processing_lock:
            self.chunk_counter += 1
            chunk_id = self.chunk_counter
            self.processing_count += 1

        # Запускаем обработку В ФОНЕ (асинхронно, БЕЗ ОЖИДАНИЯ)
        import time
        current_time = time.strftime('%H:%M:%S')
        self.logger.info(
            f"\n╔═══ CHUNK #{chunk_id} QUEUED [{current_time}] ═══╗\n"
            f"║ Processing: {self.processing_count}\n"
            f"║ Ready queue: {self.ready_queue.qsize()}\n"
            f"╚{'═' * 40}╝"
        )
        asyncio.create_task(self._process_batch_async(audio_array, chunk_id))

    def _calculate_adaptive_speed(self, queue_size: int) -> float:
        """
        Вычисляет adaptive TTS speed на основе размера очереди.

        Логика: Если очередь растёт → увеличить скорость чтобы не накапливалось отставание.

        Args:
            queue_size: Количество чанков в ready_queue

        Returns:
            float: Скорость TTS (1.0 = normal, 2.0 = 2x faster)

        Примеры:
            queue_size=0-2 → min_speed (1.3x)
            queue_size=5+  → max_speed (2.0x)
            queue_size=3-4 → linear interpolation между min и max
        """
        config = self.adaptive_speed_config
        min_speed = config.get("min_speed", 1.3)
        max_speed = config.get("max_speed", 2.0)
        low_threshold = config.get("queue_threshold_low", 2)
        high_threshold = config.get("queue_threshold_high", 5)

        if queue_size <= low_threshold:
            return min_speed
        elif queue_size >= high_threshold:
            return max_speed
        else:
            # Linear interpolation between min and max
            ratio = (queue_size - low_threshold) / (high_threshold - low_threshold)
            return min_speed + (max_speed - min_speed) * ratio

    async def _process_batch_async(self, audio_array: np.ndarray, chunk_id: int) -> None:
        """
        Фоновая обработка батча через полный pipeline.

        Обрабатывает батч (STT → LLM → TTS) и кладет результат в ready_queue.
        Выполняется полностью асинхронно, не блокируя другие батчи.

        КОНВЕЙЕРНАЯ ОБРАБОТКА:
        - Батч проходит через этапы: STT → Translation → TTS
        - Каждый этап обрабатывает только 1 батч за раз (Semaphore)
        - Разные батчи могут быть на разных этапах одновременно

        Args:
            audio_array: Numpy массив с аудио (float32, 16kHz)
            chunk_id: Уникальный ID чанка для отслеживания
        """
        try:
            # Обрабатываем батч через полный pipeline (5-10 секунд)
            # process_batch использует пошаговые semaphores внутри
            processed = await self.process_batch(audio_array, chunk_id)

            # Если process_batch вернул None (напр. русская речь) - пропускаем
            if processed is None:
                self.logger.debug("Batch processing returned None (skipped) - releasing semaphore")
                self.pipeline_semaphore.release()  # Освобождаем слот сразу
                return

            # Помечаем, что этот батч захватил pipeline_semaphore
            # (нужно освободить после воспроизведения)
            processed['_pipeline_semaphore_acquired'] = True

            # SEQUENTIAL PLAYBACK: Add chunk to buffer by chunk_id
            chunk_id = processed.get('chunk_id', 0)
            self.completed_chunks_buffer[chunk_id] = processed

            # Log if chunk completed out of order (will wait in buffer)
            if chunk_id != self.next_playback_chunk_id:
                buffer_size = len(self.completed_chunks_buffer)
                self.logger.info(
                    f"🔄 Chunk #{chunk_id} completed OUT OF ORDER "
                    f"(waiting for #{self.next_playback_chunk_id}). "
                    f"Buffer: {buffer_size} chunks waiting"
                )

            # Check if we can play next sequential chunks
            # (This moves chunks from buffer to ready_queue in sequential order)
            while self.next_playback_chunk_id in self.completed_chunks_buffer:
                next_batch = self.completed_chunks_buffer.pop(self.next_playback_chunk_id)

                # QUEUE OVERFLOW PROTECTION
                current_queue_size = self.ready_queue.qsize()
                if current_queue_size >= self.max_ready_queue_size:
                    # Skip old chunk
                    try:
                        old_batch = self.ready_queue.get_nowait()
                        self.ready_queue.task_done()
                        if old_batch.get('_pipeline_semaphore_acquired'):
                            self.pipeline_semaphore.release()
                        self.logger.warning(
                            f"⚠️ Queue OVERFLOW ({current_queue_size}/{self.max_ready_queue_size}) - "
                            f"SKIPPED old chunk (duration: {old_batch.get('duration', 0):.1f}s)"
                        )
                    except asyncio.QueueEmpty:
                        pass

                # Add to ready_queue in sequential order
                await self.ready_queue.put(next_batch)
                self.next_playback_chunk_id += 1

                self.logger.debug(f"Chunk #{chunk_id} added to ready_queue (sequential order)")

            self.logger.debug("Batch processed and queued for playback")

        except Exception as e:
            self.logger.error(f"Background batch processing failed: {e}")
            self.metrics.record_error("batch_processing_async", str(e))

            # При ошибке ОСВОБОЖДАЕМ semaphore сразу (батч не дойдет до playback)
            self.pipeline_semaphore.release()

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
        Внутренний цикл воспроизведения с буферизацией для NON-STOP режима.

        ЛОГИКА NON-STOP:
        1. Ждёт накопления min_ready_chunks_before_start готовых чанков (2-3 шт)
        2. Начинает воспроизведение только когда есть буфер
        3. Дальше играет non-stop из очереди

        Это гарантирует что пока играет один чанк, следующий уже готов!
        """
        self.logger.info("Playback loop running")

        while self.is_running:
            try:
                # КРИТИЧНО: Первый запуск - ждём накопления буфера!
                if not self.playback_started:
                    # Ждём пока накопится минимум чанков
                    while self.ready_queue.qsize() < self.min_ready_chunks_before_start:
                        current_ready = self.ready_queue.qsize()
                        self.logger.info(
                            f"🔄 Buffering before playback start: "
                            f"{current_ready}/{self.min_ready_chunks_before_start} chunks ready, "
                            f"{self.processing_count} processing..."
                        )
                        # No timeout, no keepalive spam - user controls disconnect via Stop button
                        await asyncio.sleep(0.5)  # Проверяем каждые 0.5 сек

                    self.playback_started = True
                    self.logger.info(
                        f"🚀 BUFFER READY! Starting NON-STOP playback with "
                        f"{self.ready_queue.qsize()} chunks buffered"
                    )

                # Берем следующий готовый батч из очереди (ждем если пусто)
                batch = await self.ready_queue.get()

                # Логируем состояние очереди
                queue_size = self.ready_queue.qsize()
                buffer_size = len(self.completed_chunks_buffer)
                if queue_size == 0:
                    self.logger.warning(
                        f"⚠️ Queue EMPTY during playback! "
                        f"Processing: {self.processing_count}, "
                        f"Buffer: {buffer_size} chunks waiting for order. May cause gaps!"
                    )

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

        После воспроизведения ОСВОБОЖДАЕТ слот в pipeline (pipeline_semaphore),
        позволяя следующему батчу начать обработку.

        Args:
            batch: Обработанный батч с полями:
                - original: английский текст
                - translated: русский перевод
                - audio: WAV байты
                - duration: длительность аудио
        """
        try:
            chunk_id = batch.get('chunk_id', '?')
            queue_size = self.ready_queue.qsize()
            current_time = time.strftime('%H:%M:%S')

            self.logger.info(
                f"\n╔═══ ▶️  PLAYBACK START [{current_time}] ═══╗\n"
                f"║ Chunk #{chunk_id}\n"
                f"║ Duration: {batch['duration']:.1f}s\n"
                f"║ Queue: {queue_size} chunks waiting\n"
                f"╚{'═' * 40}╝"
            )

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

            # Отправляем обновленные метрики
            metrics_data = self.metrics.get_summary()
            # Add latency fields compatible with UI expected format:
            # UI expects metrics.latency.stt, etc. Our get_summary returns latency_avg dict.
            # We should map it to the structure the UI expects or ensure UI reads from latency_avg.
            # Looking at app.js: metrics.latency.stt. So we need to ensure metrics_data['latency'] is that dict.
            # metrics.get_summary() returns "latency_avg" key.
            # Let's map it for UI compatibility.
            ui_metrics = {
                "type": "metrics",
                "data": {
                    "latency": metrics_data["latency_avg"],
                    "batches_processed": self.metrics.batches_processed,
                    "uptime": metrics_data["session_duration"],
                    "slots": self.get_status()["slots"]
                }
            }
            await self.websocket.send_json(ui_metrics)

            # Ждём окончания воспроизведения
            await asyncio.sleep(batch["duration"])

            # Увеличиваем счётчик обработанных батчей
            self.metrics.batches_processed += 1

            # FINAL METRICS UPDATE (to show updated batch count)
            final_metrics = self.metrics.get_summary()
            await self.websocket.send_json({
                "type": "metrics",
                "data": {
                    "latency": final_metrics["latency_avg"],
                    "batches_processed": self.metrics.batches_processed,
                    "uptime": final_metrics["session_duration"],
                    "slots": self.get_status()["slots"]
                }
            })

            playback_done_time = time.strftime('%H:%M:%S')
            self.logger.info(
                f"\n╔═══ ✅ PLAYBACK DONE [{playback_done_time}] ═══╗\n"
                f"║ Chunk #{chunk_id} played ({batch['duration']:.1f}s)\n"
                f"║ Total processed: {self.metrics.batches_processed}\n"
                f"╚{'═' * 40}╝"
            )

        finally:
            # КРИТИЧНО: Освобождаем слот в pipeline после воспроизведения
            # Это позволяет следующему батчу начать обработку (non-stop конвейер)
            if batch.get('_pipeline_semaphore_acquired'):
                self.pipeline_semaphore.release()
                slots_available = self.pipeline_semaphore._value
                self.logger.info(f"✅ Pipeline slot released (available: {slots_available}/{self.max_concurrent_batches}, processing: {self.processing_count}, ready: {self.ready_queue.qsize()})")

    async def stop_playback_loop(self) -> None:
        """
        Останавливает фоновый цикл воспроизведения.

        Должен быть вызван при остановке сессии.
        """
        self.is_running = False
        self.playback_started = False  # Сбрасываем флаг для следующего запуска

        if self.playback_task:
            self.playback_task.cancel()
            try:
                await self.playback_task
            except asyncio.CancelledError:
                pass

        # КРИТИЧНО: Очищаем sequential buffer между сессиями
        # (иначе chunks из старой сессии будут ждать в buffer)
        self.completed_chunks_buffer.clear()
        self.logger.debug("Sequential buffer cleared for next session")

        self.logger.info("Playback loop stopped")

    def shutdown(self) -> None:
        """
        Останавливает все ресурсы (НЕ включая preloaded TTS worker pool).

        Preloaded TTS worker pool остаётся живым между сессиями!
        Должен быть вызван при завершении сессии.
        """
        self.logger.info("Shutting down BatchQueue...")

        # КРИТИЧНО: НЕ останавливаем PRELOADED worker pool (он shared между сессиями!)
        # Останавливаем только если создали сами (не preloaded)
        if self.tts_worker_pool and not self._tts_is_preloaded:
            self.logger.info("Shutting down session-specific TTS worker pool...")
            self.tts_worker_pool.shutdown()
        elif self.tts_worker_pool and self._tts_is_preloaded:
            self.logger.info("Keeping PRELOADED TTS worker pool alive for next session ✓")

        self.logger.info("BatchQueue shutdown complete")

    async def process_batch(self, audio_array: np.ndarray, chunk_id: int) -> Dict[str, Any]:
        """
        Обрабатывает батч через STT → LLM → TTS pipeline с конвейерной обработкой.

        КОНВЕЙЕРНАЯ АРХИТЕКТУРА:
        - STEP 1 (STT): Только 1 батч за раз через whisper_semaphore
        - STEP 2 (Translation): Только 1 батч за раз через translation_semaphore
        - STEP 3 (TTS): Только 1 батч за раз через tts_semaphore

        Это позволяет разным батчам быть на разных этапах одновременно:
        - Батч #1: TTS
        - Батч #2: Translation
        - Батч #3: STT

        Args:
            audio_array: Аудио массив (float32, 16kHz)
            chunk_id: Уникальный ID чанка для отслеживания

        Returns:
            Dict с результатами обработки
        """
        try:
            pipeline_start = time.time()

            # STEP 1: STT (Local Whisper on GPU or Groq)
            async with self.whisper_semaphore:
                start = time.time()
                self.logger.info(f"🎧 Chunk #{chunk_id} → WHISPER...")
                transcription = await self.whisper_client.transcribe(audio_array)
                stt_duration = time.time() - start
                self.metrics.record_latency("stt", stt_duration)
                self.logger.info(f"   ✅ WHISPER done: {stt_duration:.2f}s → \"{transcription['text'][:60]}...\"")

            # БЛОКИРОВКА НЕ-АНГЛИЙСКОЙ РЕЧИ: Переводим ТОЛЬКО английский, всё остальное игнорируем
            # (Пример: разговор с женой на русском не должен попадать в переводчик)
            detected_lang = transcription.get("language", "unknown").lower()
            if detected_lang not in ["en", "english"]:
                self.logger.warning(
                    f"⛔ Non-English speech detected ({detected_lang}) - SKIPPING translation: "
                    f"'{transcription['text'][:50]}...'"
                )

                # Отправляем уведомление в UI
                await self.websocket.send_json({
                    "type": "non_english_detected",
                    "text": transcription["text"],
                    "language": detected_lang,
                    "message": f"Non-English speech detected ({detected_lang}) - translation skipped",
                    "timestamp": time.time()
                })

                # Завершаем обработку - не переводим, не озвучиваем
                return None

            # STEP 2: Translation (OpenRouter + context + topic)
            async with self.translation_semaphore:
                start = time.time()
                self.logger.info(f"🌐 Chunk #{chunk_id} → TRANSLATION...")
                context = await self.context_buffer.get_context()
                translation = await self.openrouter_client.translate(
                    transcription["text"], context, topic=self.topic
                )
                translation_duration = time.time() - start
                self.metrics.record_latency("translation", translation_duration)
                self.logger.info(f"   ✅ TRANSLATION done: {translation_duration:.2f}s → \"{translation[:60]}...\"")

                # Добавляем в контекст (для следующих переводов)
                await self.context_buffer.add_sentence(transcription["text"])

            # STEP 3: TTS (Worker Pool - parallel processing on 2 GPUs)
            start = time.time()
            self.logger.info(f"🔊 Chunk #{chunk_id} → TTS (dispatching to worker pool)...")

            # Выбираем метод синтеза (worker pool или legacy direct engine)
            if self.tts_worker_pool:
                audio_bytes = await self.tts_worker_pool.synthesize(translation)
            else:
                audio_bytes = await self.xtts_engine.synthesize(translation)

            tts_duration = time.time() - start
            self.metrics.record_latency("tts", tts_duration)
            self.logger.info(f"   ✅ TTS done: {tts_duration:.2f}s")

            # Вычисляем длительность аудио из WAV header
            # Получаем sample rate из конфига (worker pool не имеет direct access к engine)
            tts_config = load_config()["models"]["tts"]
            tts_sample_rate = tts_config.get("output_sample_rate", 24000)
            audio_data_size = len(audio_bytes) - 44  # Subtract WAV header
            audio_duration = audio_data_size / (tts_sample_rate * 2)  # 2 bytes/sample (int16)

            # E2E метрика
            e2e_duration = stt_duration + translation_duration + tts_duration
            pipeline_total = time.time() - pipeline_start
            self.metrics.record_latency("e2e", e2e_duration)

            # Detailed pipeline summary
            audio_samples = (len(audio_bytes) - 44) / 2  # int16 samples
            input_duration = len(audio_array) / 16000  # Input audio duration
            self.logger.info(
                f"\n╔═══ CHUNK #{chunk_id} PIPELINE COMPLETE ═══╗\n"
                f"║ INPUT:  {input_duration:.1f}s audio\n"
                f"║ OUTPUT: {audio_duration:.1f}s TTS ({int(audio_samples)} samples)\n"
                f"║ ─────────────────────────────────────\n"
                f"║ WHISPER:     {stt_duration:6.2f}s\n"
                f"║ TRANSLATION: {translation_duration:6.2f}s\n"
                f"║ TTS:         {tts_duration:6.2f}s\n"
                f"║ ─────────────────────────────────────\n"
                f"║ E2E TOTAL:   {e2e_duration:6.2f}s\n"
                f"║ Ready queue: {self.ready_queue.qsize()} chunks\n"
                f"╚{'═' * 40}╝"
            )

            # Возвращаем результат
            return {
                "chunk_id": chunk_id,  # Добавляем ID для отслеживания
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
            raise  # Пробрасываем выше

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
