"""
TTS Worker Pool для параллельной обработки на нескольких GPU.

АРХИТЕКТУРА:
- Worker #1 → GPU 0 (вместе с Whisper)
- Worker #2 → GPU 1 (отдельный GPU)
- Multiprocessing (spawn) для избежания CUDA race conditions
- Queue-based async processing

РЕШЕНИЕ ПРОБЛЕМЫ:
XTTS-v2 не является thread-safe и вызывает CUDA assertion errors
при concurrent requests. Решение: загрузить 2 копии модели в
разных процессах на разные GPU.
"""

import os
import time
import subprocess
import multiprocessing as mp
from queue import Empty
import numpy as np
import asyncio
from app.config import load_config
from app.monitoring.logger import setup_logger


def _apply_atempo(wav_bytes: bytes, speed: float) -> bytes:
    """
    Apply time-stretch without pitch change using ffmpeg atempo filter.

    atempo accepts 0.5–2.0 per stage; chain two stages for speed > 2.0.
    Falls back to original bytes if ffmpeg is unavailable.
    """
    if speed <= 1.001:
        return wav_bytes

    if speed <= 2.0:
        filter_str = f"atempo={speed:.4f}"
    else:
        # e.g. 4.0 → atempo=2.0,atempo=2.0
        filter_str = f"atempo=2.0,atempo={speed / 2.0:.4f}"

    cmd = [
        "ffmpeg", "-y",
        "-f", "wav", "-i", "pipe:0",
        "-filter:a", filter_str,
        "-f", "wav", "pipe:1"
    ]
    try:
        result = subprocess.run(cmd, input=wav_bytes, capture_output=True, timeout=30)
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except Exception:
        pass
    return wav_bytes  # fallback: return original unchanged


def _tts_worker_process(gpu_id: int, input_queue: mp.Queue, output_queue: mp.Queue, ready_event: mp.Event):
    """
    Worker процесс для TTS синтеза на отдельном GPU.

    Args:
        gpu_id: ID GPU для этого worker'а (0 или 1)
        input_queue: Очередь входящих запросов (request_id, text)
        output_queue: Очередь результатов (request_id, audio_bytes, duration)
        ready_event: Event для сигнализации что worker готов
    """
    # КРИТИЧНО: Установить CUDA_VISIBLE_DEVICES ДО импорта PyTorch/TTS
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Теперь импортируем TTS (после установки GPU)
    from app.components.xtts_engine import XTTSEngine

    logger = setup_logger(f"tts_worker_{gpu_id}")
    logger.info(f"🚀 TTS Worker #{gpu_id} starting on GPU {gpu_id}...")

    try:
        # Загружаем модель на этот GPU
        # ВАЖНО: После CUDA_VISIBLE_DEVICES=gpu_id, используем device_id=0
        # (т.к. выбранный GPU становится единственным видимым с индексом 0)
        engine = XTTSEngine(device_override='cuda', gpu_id_override=0)
        logger.info(f"✅ TTS Worker #{gpu_id} READY on GPU {gpu_id}")

        # Сигнализируем что готовы
        ready_event.set()

        # Основной цикл обработки
        while True:
            try:
                # Ждём запрос из очереди (timeout чтобы можно было завершить процесс)
                request = input_queue.get(timeout=1.0)

                if request is None:  # Poison pill для завершения
                    logger.info(f"TTS Worker #{gpu_id} received shutdown signal")
                    break

                request_id, text, speed = request

                # Smart speed split: XTTS ≤ 2.0 (quality limit), ffmpeg atempo handles the rest
                # e.g. speed=4.0 → XTTS=2.0, atempo=2.0 → same total 4x, no chipmunk effect
                xtts_speed = min(speed, 2.0)
                atempo_factor = speed / xtts_speed  # e.g. 4.0/2.0 = 2.0

                engine.speed = xtts_speed

                # Обрабатываем TTS (async call в sync worker - используем asyncio.run)
                start_time = time.time()
                audio_bytes = asyncio.run(engine.synthesize(text))
                duration = time.time() - start_time

                # Вычисляем длительность аудио
                tts_sample_rate = engine.output_sample_rate
                audio_data_size = len(audio_bytes) - 44  # WAV header
                audio_duration = audio_data_size / (tts_sample_rate * 2)  # 2 bytes/sample

                # Apply ffmpeg atempo for remaining speed (pitch-preserving time-stretch)
                if atempo_factor > 1.001:
                    audio_bytes = _apply_atempo(audio_bytes, atempo_factor)
                    audio_data_size = len(audio_bytes) - 44
                    audio_duration = audio_data_size / (tts_sample_rate * 2)

                logger.info(
                    f"Worker #{gpu_id}: Request #{request_id} done in {duration:.2f}s "
                    f"(audio: {audio_duration:.1f}s, xtts_speed={xtts_speed}x, atempo={atempo_factor:.2f}x)"
                )

                # Отправляем результат
                output_queue.put((request_id, audio_bytes, audio_duration))

            except Empty:
                # Timeout - просто продолжаем цикл
                continue
            except Exception as e:
                logger.error(f"Worker #{gpu_id} processing error: {e}")
                # Отправляем ошибку в output queue
                output_queue.put((request_id, None, None))

    except Exception as e:
        logger.error(f"Worker #{gpu_id} initialization failed: {e}")
        ready_event.set()  # Всё равно сигнализируем чтобы не зависнуть

    logger.info(f"TTS Worker #{gpu_id} shutdown")


class TTSWorkerPool:
    """
    Пул TTS workers для параллельной обработки на нескольких GPU.

    Использует multiprocessing для создания worker процессов,
    каждый из которых загружает свою копию XTTS модели на свой GPU.

    Это решает проблему CUDA assertion errors при concurrent requests.
    """

    def __init__(self, num_workers: int = 2):
        """
        Инициализация worker pool.

        Args:
            num_workers: Количество worker процессов (default: 2 для 2 GPU)
        """
        self.logger = setup_logger(__name__)
        self.num_workers = num_workers

        # Используем spawn method (ОБЯЗАТЕЛЬНО для CUDA!)
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            # Уже установлен - проверяем что это spawn
            if mp.get_start_method() != 'spawn':
                self.logger.warning(f"Multiprocessing start method is {mp.get_start_method()}, but spawn is required for CUDA!")
            else:
                self.logger.debug("Multiprocessing start method already set to spawn")

        # Создаём очереди для коммуникации
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()

        # Текущая скорость (можно менять динамически через set_speed)
        tts_config = load_config()["models"]["tts"]
        self.current_speed = tts_config.get("speed", 2.0)

        # Счётчик запросов
        self.request_counter = 0
        self.pending_requests = {}  # request_id -> asyncio.Future

        # Запускаем workers
        self.workers = []
        self.ready_events = []

        for gpu_id in range(num_workers):
            ready_event = mp.Event()
            self.ready_events.append(ready_event)

            worker = mp.Process(
                target=_tts_worker_process,
                args=(gpu_id, self.input_queue, self.output_queue, ready_event),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            self.logger.info(f"Started TTS worker process on GPU {gpu_id} (PID: {worker.pid})")

        # Ждём пока все workers загрузятся
        self.logger.info(f"Waiting for {num_workers} TTS workers to initialize...")
        for i, event in enumerate(self.ready_events):
            event.wait(timeout=60)  # 60 секунд на загрузку модели
            self.logger.info(f"Worker #{i} is READY")

        self.logger.info(f"✅ TTSWorkerPool initialized with {num_workers} workers")

    async def synthesize(self, text: str) -> bytes:
        """
        Синтезирует речь (асинхронно распределяет по workers).

        Args:
            text: Текст для синтеза

        Returns:
            bytes: WAV аудио данные
        """
        # Создаём уникальный ID запроса
        self.request_counter += 1
        request_id = self.request_counter

        # Создаём Future для результата
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[request_id] = future

        # Отправляем запрос в очередь с текущей скоростью
        self.input_queue.put((request_id, text, self.current_speed))

        # Запускаем фоновую задачу для получения результата
        asyncio.create_task(self._collect_result(request_id))

        # Ждём результат
        audio_bytes, audio_duration = await future

        return audio_bytes

    async def _collect_result(self, request_id: int):
        """
        Фоновая задача для получения результата из output_queue.

        Args:
            request_id: ID запроса
        """
        # Ждём результат в output_queue (асинхронно)
        while True:
            try:
                # Проверяем output_queue (non-blocking)
                result_id, audio_bytes, audio_duration = self.output_queue.get_nowait()

                if result_id == request_id:
                    # Это наш результат!
                    future = self.pending_requests.pop(request_id)
                    future.set_result((audio_bytes, audio_duration))
                    return
                else:
                    # Не наш результат - обрабатываем другой запрос
                    if result_id in self.pending_requests:
                        other_future = self.pending_requests.pop(result_id)
                        other_future.set_result((audio_bytes, audio_duration))

            except:
                # Очередь пуста - ждём немного
                await asyncio.sleep(0.01)

    def set_speed(self, speed: float) -> None:
        """
        Динамически меняет скорость TTS.
        Применяется к следующим запросам синтеза (уже запущенные не меняются).

        Args:
            speed: Множитель скорости (1.0 = норма, 4.0 = 4x быстрее)
        """
        self.current_speed = speed
        self.logger.info(f"TTS speed changed to {speed}x")

    def shutdown(self):
        """Останавливает все worker процессы."""
        self.logger.info("Shutting down TTS worker pool...")

        # Отправляем poison pills
        for _ in range(self.num_workers):
            self.input_queue.put(None)

        # Ждём завершения
        for i, worker in enumerate(self.workers):
            worker.join(timeout=5)
            if worker.is_alive():
                self.logger.warning(f"Worker #{i} did not shut down gracefully, terminating...")
                worker.terminate()

        self.logger.info("TTS worker pool shut down")
