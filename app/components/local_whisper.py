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
        self.gpu_id = self.config.get("gpu_id", 0)  # GPU 0 для Whisper

        # Загружаем модель
        self.logger.info(f"Loading Whisper model: {self.model_size} on {self.device}:{self.gpu_id}")

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            device_index=self.gpu_id,
            compute_type=self.compute_type
        )

        self.logger.info(f"Local Whisper initialized: {self.model_size} on GPU {self.gpu_id}")

    async def transcribe(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """
        Транскрибирует аудио через локальный Whisper.

        Args:
            audio_array: Numpy массив с аудио (float32, 16kHz)

        Returns:
            dict: {"text": "...", "language": "en"}

        Алгоритм:
            1. Преобразуем numpy array в формат для faster-whisper
            2. Запускаем транскрипцию на GPU
            3. Собираем сегменты в единый текст
            4. Возвращаем результат
        """
        try:
            # faster-whisper принимает numpy array напрямую
            # Убеждаемся что это float32
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # Запускаем транскрипцию (синхронно, но через to_thread)
            segments, info = await asyncio.to_thread(
                self.model.transcribe,
                audio_array,
                language=self.language,
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
            self.logger.error(f"Local Whisper error: {e}")
            raise
