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

        self.lock = asyncio.Lock()

        # Log current VAD settings
        min_chunk = self.config["vad"]["min_chunk_duration"]
        max_chunk = self.config["vad"]["max_phrase_duration"]
        min_silence = self.config["vad"]["min_silence_duration"]
        self.logger.info(f"StreamProcessor initialized (FAT chunks: {min_chunk}-{max_chunk}s, silence: {min_silence}s)")
    
    async def process_chunk(self, audio_bytes: bytes) -> None:
        """
        Обрабатывает входящий аудио чанк.

        ЛОГИКА ЖИРНЫХ БАТЧЕЙ (FAT CHUNKS):
        - До min_chunk_duration (12s): ИГНОРИРУЕМ тишину полностью, накапливаем
        - От min_chunk до max_phrase (12-18s): ИЩЕМ тишину для логичного разрыва
        - На max_phrase_duration (18s): ПРИНУДИТЕЛЬНАЯ финализация

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
                min_chunk = self.config["vad"]["min_chunk_duration"]
                min_silence = self.config["vad"]["min_silence_duration"]
                self.logger.info(f"🎤 New phrase started (FAT chunk: {min_chunk}s min, silence: {min_silence}s)")

            # Пока фраза активна - добавляем ВСЁ аудио (и речь, и паузы)
            should_finalize = False
            phrase_to_finalize = None

            if self.phrase_start_time is not None:
                self.current_phrase.extend(audio_float)
                phrase_duration = time.time() - self.phrase_start_time

                # ВАЖНО: min_chunk и max_chunk из конфига (FAT chunks для перекрытия processing time)
                min_chunk = self.config["vad"]["min_chunk_duration"]  # 12.0 сек
                max_chunk = self.config["vad"]["max_phrase_duration"]  # 18.0 сек

                # Логируем прогресс каждые 3 секунды (для диагностики)
                if int(phrase_duration) % 3 == 0 and int(phrase_duration) > 0:
                    silence_frames = self.vad.silence_frames
                    min_silence_frames = int(self.config["vad"]["min_silence_duration"] * 10)
                    self.logger.debug(f"⏱️ Phrase progress: {phrase_duration:.1f}s (silence: {silence_frames}/{min_silence_frames} frames, min_chunk: {min_chunk}s)")

                # КРИТИЧНО: ФИНАЛИЗАЦИЯ ТОЛЬКО ЕСЛИ ФРАЗА >= min_chunk (9 сек)!
                # До 9 секунд - ИГНОРИРУЕМ ТИШИНУ ПОЛНОСТЬЮ
                if phrase_duration < min_chunk:
                    # ДО МИНИМУМА - НЕ ПРОВЕРЯЕМ ТИШИНУ, ПРОСТО НАКАПЛИВАЕМ
                    # Логируем что накапливаем (каждые 3 сек)
                    if int(phrase_duration) % 3 == 0 and int(phrase_duration) > 0:
                        self.logger.debug(f"📦 Accumulating (before min_chunk): {phrase_duration:.1f}s / {min_chunk}s")
                    pass  # should_finalize остается False

                elif phrase_duration >= max_chunk:
                    # ЖЁСТКИЙ ЛИМИТ: >= 13 сек - ПРИНУДИТЕЛЬНАЯ ФИНАЛИЗАЦИЯ
                    self.logger.warning(f"✂️ Chunk FORCED: {phrase_duration:.1f}s (max limit {max_chunk}s reached!)")
                    should_finalize = True

                else:
                    # ЗОНА ПОИСКА ТИШИНЫ: 9-13 сек
                    # Ищем логичный разрыв (тишина 1 сек)
                    if self.vad.is_silence_ready():
                        self.logger.info(f"✂️ Chunk ready: {phrase_duration:.1f}s (silence {self.config['vad']['min_silence_duration']}s detected in range {min_chunk}-{max_chunk}s)")
                        should_finalize = True
                    else:
                        # Нет тишины - продолжаем накапливать (логируем каждые 2 сек)
                        if int(phrase_duration) % 2 == 0:
                            silence_frames = self.vad.silence_frames
                            min_silence_frames = int(self.config["vad"]["min_silence_duration"] * 10)
                            self.logger.debug(f"⏳ Waiting for silence: {phrase_duration:.1f}s (silence: {silence_frames}/{min_silence_frames} frames)")

                # Подготавливаем данные для финализации ПОД БЛОКИРОВКОЙ
                if should_finalize:
                    phrase_to_finalize = self.current_phrase.copy()
                    self.current_phrase = []
                    self.phrase_start_time = None
                    self.vad.reset()  # Сбрасываем счётчики для следующего чанка

        # Финализируем ВНЕ блокировки, чтобы не блокировать входящие чанки
        if should_finalize and phrase_to_finalize:
            await self.finalize_phrase(phrase_to_finalize)
    
    async def finalize_phrase(self, phrase_data: list) -> None:
        """
        Финализирует фразу ВНЕ блокировки для неблокирующей обработки.

        Args:
            phrase_data: Список аудио семплов для финализации
        """
        min_samples = int(self.vad.min_speech_duration * self.sample_rate)

        if len(phrase_data) < min_samples:
            self.logger.debug(f"Phrase too short: {len(phrase_data)/self.sample_rate:.1f}s < {self.vad.min_speech_duration}s, skipping")
            return

        phrase_array = np.array(phrase_data, dtype=np.float32)
        phrase_array = normalize_audio(phrase_array)

        duration = len(phrase_array) / self.sample_rate
        self.logger.info(f"=== CHUNK FINALIZED: {duration:.1f}s ({len(phrase_array)} samples) ===")
        await self.batch_queue.add_batch(phrase_array)
