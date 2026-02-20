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

        if tts_engine:
            self.xtts_engine = tts_engine
            self.logger.info("Using preloaded TTS engine")
        else:
            self.xtts_engine = XTTSEngine()

        self.context_buffer = ContextBuffer()

        # Topic/context for better translation accuracy
        self.topic = topic

        # НОВАЯ АРХИТЕКТУРА: Очередь готовых батчей (FIFO)
        self.ready_queue = asyncio.Queue()  # Неограниченная очередь готовых батчей
        self.playback_task = None  # Фоновая задача воспроизведения
        self.is_running = False

        # NON-STOP PLAYBACK: Минимум готовых чанков перед стартом воспроизведения
        # Это критично для дословного режима (маленькие чанки)
        self.min_ready_chunks_before_start = self.config.get("min_ready_chunks_before_start", 3)
        self.playback_started = False  # Флаг первого запуска

        # Счетчики для мониторинга
        self.processing_count = 0  # Сколько батчей сейчас обрабатывается
        self.processing_lock = asyncio.Lock()  # Для атомарности счетчика

        # PIPELINE CONCURRENCY CONTROL (каждый этап обрабатывает только 1 батч за раз)
        # Это позволяет разным батчам быть на разных этапах одновременно (конвейер)
        self.whisper_semaphore = asyncio.Semaphore(1)  # Только 1 батч в STT
        self.translation_semaphore = asyncio.Semaphore(1)  # Только 1 батч в Translation
        self.tts_semaphore = asyncio.Semaphore(1)  # Только 1 батч в TTS

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

        # Увеличиваем счетчик обрабатываемых батчей
        async with self.processing_lock:
            self.processing_count += 1

        # Запускаем обработку В ФОНЕ (асинхронно, БЕЗ ОЖИДАНИЯ)
        asyncio.create_task(self._process_batch_async(audio_array))

        self.logger.debug(f"Batch queued for processing (total processing: {self.processing_count}, pipeline slots: {self.pipeline_semaphore._value}/{self.max_concurrent_batches})")

    async def _process_batch_async(self, audio_array: np.ndarray) -> None:
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
        """
        try:
            # Обрабатываем батч через полный pipeline (5-10 секунд)
            # process_batch использует пошаговые semaphores внутри
            processed = await self.process_batch(audio_array)

            # Если process_batch вернул None (напр. русская речь) - пропускаем
            if processed is None:
                self.logger.debug("Batch processing returned None (skipped) - releasing semaphore")
                self.pipeline_semaphore.release()  # Освобождаем слот сразу
                return

            # Помечаем, что этот батч захватил pipeline_semaphore
            # (нужно освободить после воспроизведения)
            processed['_pipeline_semaphore_acquired'] = True

            # Кладем готовый батч в очередь воспроизведения
            await self.ready_queue.put(processed)

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
                if queue_size == 0:
                    self.logger.warning(
                        f"⚠️ Queue EMPTY during playback! "
                        f"Processing: {self.processing_count}. May cause gaps!"
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

            self.logger.info(f"=== BATCH #{batch_num} DONE (played {batch['duration']:.1f}s) ===")

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

        self.logger.info("Playback loop stopped")

    async def process_batch(self, audio_array: np.ndarray) -> Dict[str, Any]:
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

        Returns:
            Dict с результатами обработки
        """
        try:
            # STEP 1: STT (Local Whisper on GPU or Groq)
            # Только 1 батч может использовать Whisper одновременно
            async with self.whisper_semaphore:
                start = time.time()
                transcription = await self.whisper_client.transcribe(audio_array)
                stt_duration = time.time() - start
                self.metrics.record_latency("stt", stt_duration)
                self.logger.debug(f"STT completed in {stt_duration:.2f}s")

            # БЛОКИРОВКА РУССКОЙ РЕЧИ: Если обнаружен русский язык - пропускаем перевод
            detected_lang = transcription.get("language", "unknown").lower()
            if detected_lang in ["ru", "russian", "rus"]:
                self.logger.warning(f"⛔ Russian speech detected - SKIPPING translation: '{transcription['text'][:50]}...'")

                # Отправляем уведомление в UI
                await self.websocket.send_json({
                    "type": "russian_detected",
                    "text": transcription["text"],
                    "message": "Russian speech detected - translation skipped",
                    "timestamp": time.time()
                })

                # Завершаем обработку - не переводим, не озвучиваем
                return None

            # STEP 2: Translation (OpenRouter + context + topic)
            # Только 1 батч может переводить одновременно
            async with self.translation_semaphore:
                start = time.time()
                context = await self.context_buffer.get_context()
                translation = await self.openrouter_client.translate(
                    transcription["text"], context, topic=self.topic
                )
                translation_duration = time.time() - start
                self.metrics.record_latency("translation", translation_duration)
                self.logger.debug(f"Translation completed in {translation_duration:.2f}s")

                # Добавляем в контекст (для следующих переводов)
                await self.context_buffer.add_sentence(transcription["text"])

            # STEP 3: TTS (XTTS-v2)
            # Только 1 батч может синтезировать речь одновременно
            async with self.tts_semaphore:
                start = time.time()
                audio_bytes = await self.xtts_engine.synthesize(translation)
                tts_duration = time.time() - start
                self.metrics.record_latency("tts", tts_duration)
                self.logger.debug(f"TTS completed in {tts_duration:.2f}s")

            # Вычисляем длительность аудио из WAV header
            tts_sample_rate = self.xtts_engine.output_sample_rate
            audio_data_size = len(audio_bytes) - 44  # Subtract WAV header
            audio_duration = audio_data_size / (tts_sample_rate * 2)  # 2 bytes/sample (int16)

            # E2E метрика
            e2e_duration = stt_duration + translation_duration + tts_duration
            self.metrics.record_latency("e2e", e2e_duration)

            # Логируем успешную обработку
            log_json(self.logger, "INFO", "Batch processed (pipeline)",
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
