"""
Модуль для работы с локальным Whisper на GPU (Speech-to-Text).
Использует faster-whisper для быстрой транскрибации на CUDA.
"""

import asyncio
import numpy as np
from faster_whisper import WhisperModel
from typing import Dict, Any
from app.config import load_config
from app.monitoring.logger import setup_logger


class LocalWhisperClient:
    """
    Клиент для локального Whisper (faster-whisper на GPU).

    Функционал:
        - Транскрибирует аудио локально на GPU
        - Поддержка CUDA с указанием конкретного GPU
        - Быстрая обработка без API лимитов
    """

    def __init__(self):
        """Инициализация клиента."""
        self.config = load_config()["models"]["whisper"]
        self.logger = setup_logger(__name__)

        # Параметры
        self.model_size = self.config.get("model_size", "large-v3")
        self.device = self.config.get("device", "cuda")
        self.compute_type = self.config.get("compute_type", "float16")
        self.language = self.config.get("language", "en")
        self.gpu_id = self.config.get("gpu_id", 0)

        # Force device_index=0 for CPU to avoid confusion
        if self.device == "cpu":
            self.gpu_id = 0

        # ВАЖНО: Mutex для CUDA операций (предотвращает race condition)
        self._cuda_lock = asyncio.Lock()

        # Загружаем модель
        self.logger.info(f"Loading Whisper model: {self.model_size} on {self.device}:{self.gpu_id} (compute: {self.compute_type})")

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            device_index=self.gpu_id,
            compute_type=self.compute_type
        )

        self.logger.info(f"Local Whisper initialized: {self.model_size} on {self.device} (with CUDA mutex)")

    async def transcribe(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """
        Транскрибирует аудио через локальный Whisper с retry логикой.

        Args:
            audio_array: Numpy массив с аудио (float32, 16kHz)

        Returns:
            dict: {"text": "...", "language": "en"}

        Алгоритм:
            1. Преобразуем numpy array в формат для faster-whisper
            2. Ждем освобождения CUDA lock (очередь)
            3. Запускаем транскрипцию на GPU (монопольно)
            4. При CUDA ошибке - retry с очисткой cache
            5. Возвращаем результат
        """
        # faster-whisper принимает numpy array напрямую
        # Убеждаемся что это float32
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Retry параметры
        max_attempts = 3
        backoff_factor = 1.5

        # КРИТИЧНО: Используем mutex для CUDA операций
        # Только один батч за раз может использовать GPU
        async with self._cuda_lock:
            for attempt in range(max_attempts):
                try:
                    # Запускаем транскрипцию (синхронно, но через to_thread)
                    # language=None для автоматического определения языка (нужно для блокировки русского)
                    segments, info = await asyncio.to_thread(
                        self.model.transcribe,
                        audio_array,
                        language=None,  # Auto-detect language (was: self.language = "en")
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(
                            threshold=0.5,
                            min_speech_duration_ms=250,
                            min_silence_duration_ms=100
                        )
                    )

                    # Собираем текст из сегментов
                    text_parts = []
                    for segment in segments:
                        text_parts.append(segment.text.strip())

                    full_text = " ".join(text_parts).strip()

                    result = {
                        "text": full_text,
                        "language": info.language
                    }

                    self.logger.info(f"Transcribed: {len(result['text'])} chars (lang: {info.language})")
                    return result

                except Exception as e:
                    error_msg = str(e).lower()
                    is_cuda_error = "cuda" in error_msg or "gpu" in error_msg

                    if is_cuda_error and attempt < max_attempts - 1:
                        wait_time = backoff_factor ** attempt
                        self.logger.warning(
                            f"CUDA error (attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"Clearing CUDA cache and retrying in {wait_time:.1f}s..."
                        )

                        # Очищаем CUDA cache
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                # Синхронизируем GPU для устранения race conditions
                                torch.cuda.synchronize(self.gpu_id)
                                self.logger.debug(f"CUDA cache cleared on GPU {self.gpu_id}")
                        except Exception as cache_error:
                            self.logger.warning(f"Failed to clear CUDA cache: {cache_error}")

                        await asyncio.sleep(wait_time)
                    else:
                        self.logger.error(f"Local Whisper error after {attempt + 1} attempts: {e}")
                        raise
