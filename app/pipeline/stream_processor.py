"""
Модуль обработки входящего аудио потока.
"""

import numpy as np
import asyncio
import time
from collections import deque
from app.config import load_config
from app.monitoring.logger import setup_logger
from app.components.audio_utils import int16_to_float32, normalize_audio
from app.components.vad_detector import VADDetector
from app.pipeline.batch_queue import BatchQueue


class StreamProcessor:
    """
    Обрабатывает входящий аудио поток от WebSocket.
    """
    
    def __init__(self, batch_queue: BatchQueue):
        """Инициализация процессора."""
        self.config = load_config()["pipeline"]
        self.logger = setup_logger(__name__)
        self.batch_queue = batch_queue
        
        self.vad = VADDetector()
        self.sample_rate = self.config["audio"]["sample_rate"]
        
        buffer_size = self.sample_rate * self.config["audio"]["buffer_duration_sec"]
        self.audio_buffer = deque(maxlen=buffer_size)
        self.current_phrase = []
        self.phrase_start_time = None

        # Max phrase duration (auto-finalize long phrases)
        self.max_phrase_duration = self.config["vad"].get("max_phrase_duration", 15.0)

        self.lock = asyncio.Lock()
        self.logger.info("StreamProcessor initialized")
    
    async def process_chunk(self, audio_bytes: bytes) -> None:
        """
        Обрабатывает входящий аудио чанк.

        ГИБРИДНАЯ ЛОГИКА:
        - Минимум 10 секунд (min_chunk_duration) - ОБЯЗАТЕЛЬНО набираем
        - После 10 сек ждём тишину для логичного разрыва
        - Максимум 15 сек (жёсткий лимит)

        ВАЖНО: Пока фраза активна, добавляем ВСЁ аудио (и речь, и паузы),
        чтобы не терять контекст и не резать на микро-паузах.
        """
        async with self.lock:
            # Конвертируем bytes → numpy
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = int16_to_float32(audio_int16)

            self.audio_buffer.extend(audio_float)

            # VAD детекция (обновляет счётчики speech_frames/silence_frames)
            is_speech = self.vad.detect_speech(audio_float)

            # Начало новой фразы - первая речь
            if is_speech and self.phrase_start_time is None:
                self.phrase_start_time = time.time()
                self.logger.debug("New phrase started")

            # Пока фраза активна - добавляем ВСЁ аудио (и речь, и паузы)
            if self.phrase_start_time is not None:
                self.current_phrase.extend(audio_float)
                phrase_duration = time.time() - self.phrase_start_time

                # ГИБРИД: После минимума ищем тишину для логичного разрыва
                min_chunk = self.config["vad"].get("min_chunk_duration", 10.0)
                max_chunk = self.max_phrase_duration  # Жёсткий лимит (15 сек)

                if phrase_duration >= min_chunk:
                    # После минимума — ищем тишину (1.5 сек паузы)
                    if self.vad.is_silence_ready():
                        self.logger.info(f"Chunk ready: {phrase_duration:.1f}s (silence detected)")
                        await self.finalize_phrase()
                        self.current_phrase = []
                        self.phrase_start_time = None
                        self.vad.reset()  # Сбрасываем счётчики для следующего чанка
                    # Жёсткий лимит — режем даже без тишины
                    elif phrase_duration >= max_chunk:
                        self.logger.info(f"Chunk forced: {phrase_duration:.1f}s (max limit)")
                        await self.finalize_phrase()
                        self.current_phrase = []
                        self.phrase_start_time = None
                        self.vad.reset()  # Сбрасываем счётчики для следующего чанка
    
    async def finalize_phrase(self) -> None:
        """
        Финализирует накопленную фразу.
        """
        min_samples = int(self.vad.min_speech_duration * self.sample_rate)

        if len(self.current_phrase) < min_samples:
            self.logger.debug(f"Phrase too short: {len(self.current_phrase)/self.sample_rate:.1f}s < {self.vad.min_speech_duration}s, skipping")
            return

        phrase_array = np.array(self.current_phrase, dtype=np.float32)
        phrase_array = normalize_audio(phrase_array)

        duration = len(phrase_array) / self.sample_rate
        self.logger.info(f"=== CHUNK FINALIZED: {duration:.1f}s ({len(phrase_array)} samples) ===")
        await self.batch_queue.add_batch(phrase_array)
