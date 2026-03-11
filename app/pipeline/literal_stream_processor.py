"""
Модуль быстрой обработки аудио потока для дословного перевода.

РЕЖИМ ДОСЛОВНОГО ПЕРЕВОДА (LITERAL MODE):
- Быстрая обработка с минимальной задержкой (<5 сек)
- Маленькие чанки (2-5 сек) для быстроты
- Короткие паузы (0.2 сек) для разделения фраз
- Контекст для точности перевода и фильтрации галлюцинаций
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


class LiteralStreamProcessor:
    """
    Быстрый процессор для дословного перевода.

    ОТЛИЧИЯ ОТ StreamProcessor:
    - Минимальная длина чанка: 2 сек (вместо 12 сек)
    - Максимальная длина чанка: 5 сек (вместо 18 сек)
    - Пауза для разрыва: 0.2 сек (вместо 1.0 сек)
    - Цель: максимальная скорость, минимальная задержка
    """

    def __init__(self, batch_queue: BatchQueue):
        """Инициализация быстрого процессора."""
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

        # ПАРАМЕТРЫ БЫСТРОГО РЕЖИМА
        self.min_chunk_duration = 1.0  # Минимум 1.0 секунды (было: 1.5)
        self.max_chunk_duration = 4.0  # Максимум 4 секунды - ПРИНУДИТЕЛЬНОЕ закрытие (было: 3.0)
        self.min_silence_duration = 0.5  # 0.5 секунды тишины = закрыть чанк (было: 1.0)

        self.logger.info(
            f"LiteralStreamProcessor: chunks {self.min_chunk_duration}-{self.max_chunk_duration}s, "
            f"silence {self.min_silence_duration}s"
        )

    async def process_chunk(self, audio_bytes: bytes) -> None:
        """
        Обрабатывает входящий аудио чанк в БЫСТРОМ РЕЖИМЕ.

        ЛОГИКА БЫСТРЫХ ЧАНКОВ:
        - До min_chunk (2s): НАКАПЛИВАЕМ (игнорируем паузы)
        - От min_chunk до max_chunk (2-5s): ИЩЕМ паузу 0.2s для разрыва
        - На max_chunk (5s): ПРИНУДИТЕЛЬНАЯ финализация

        КРИТИЧНО: Реакция на короткие паузы (0.2s) для быстрого реагирования.
        """
        async with self.lock:
            # Конвертируем bytes → numpy
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = int16_to_float32(audio_int16)

            self.audio_buffer.extend(audio_float)

            # VAD детекция
            is_speech = self.vad.detect_speech(audio_float)

            # Начало новой фразы
            if is_speech and self.phrase_start_time is None:
                self.phrase_start_time = time.time()
                self.logger.debug(
                    f"🎤 New phrase (max {self.max_chunk_duration}s, "
                    f"forced closure at {self.max_chunk_duration}s)"
                )

            should_finalize = False
            phrase_to_finalize = None

            if self.phrase_start_time is not None:
                self.current_phrase.extend(audio_float)
                phrase_duration = time.time() - self.phrase_start_time

                # БЫСТРАЯ ОБРАБОТКА: меньшие чанки, короткие паузы
                if phrase_duration < self.min_chunk_duration:
                    # ДО МИНИМУМА - НАКАПЛИВАЕМ
                    pass

                elif phrase_duration >= self.max_chunk_duration:
                    # МАКС ЛИМИТ - ПРИНУДИТЕЛЬНАЯ ФИНАЛИЗАЦИЯ
                    self.logger.info(
                        f"✂️ FAST chunk FORCED: {phrase_duration:.1f}s "
                        f"(max {self.max_chunk_duration}s)"
                    )
                    should_finalize = True

                # EMERGENCY STOP: Если чанк стал слишком большим (> 10s), финализировать немедленно
                # (Защита от "жирных чанков" из-за багов VAD или непрерывной речи)
                if phrase_duration >= 10.0:
                    self.logger.warning(
                        f"⚠️ EMERGENCY: Fat chunk detected ({phrase_duration:.1f}s) - "
                        f"forcing finalization NOW!"
                    )
                    should_finalize = True

                else:
                    # ЗОНА ПОИСКА КОРОТКОЙ ПАУЗЫ (2-5s)
                    # Проверяем паузу 0.2 сек (быстрая реакция)
                    silence_frames = self.vad.silence_frames
                    min_silence_frames = int(self.min_silence_duration * 10)  # 0.2s * 10 = 2 frames

                    if silence_frames >= min_silence_frames:
                        self.logger.info(
                            f"✂️ FAST chunk ready: {phrase_duration:.1f}s "
                            f"(pause {self.min_silence_duration}s detected)"
                        )
                        should_finalize = True

                # Финализация
                if should_finalize:
                    phrase_to_finalize = self.current_phrase.copy()
                    self.current_phrase = []
                    self.phrase_start_time = None
                    self.vad.reset()

        # Финализируем ВНЕ блокировки
        if should_finalize and phrase_to_finalize:
            await self.finalize_phrase(phrase_to_finalize)

    async def finalize_phrase(self, phrase_data: list) -> None:
        """
        Финализирует фразу и отправляет на обработку.

        Args:
            phrase_data: Список аудио семплов
        """
        import time
        finalize_start = time.time()

        # КРИТИЧНО: Используем self.min_chunk_duration (1.5s) вместо self.vad.min_speech_duration (3.0s)
        # Иначе чанки < 3.0s будут отбрасываться!
        min_samples = int(self.min_chunk_duration * self.sample_rate)

        if len(phrase_data) < min_samples:
            self.logger.debug(
                f"Phrase too short: {len(phrase_data)/self.sample_rate:.1f}s, skipping"
            )
            return

        phrase_array = np.array(phrase_data, dtype=np.float32)
        phrase_array = normalize_audio(phrase_array)

        duration = len(phrase_array) / self.sample_rate

        # Detailed finalization log
        current_time = time.strftime('%H:%M:%S')
        self.logger.info(
            f"\n╔═══ CHUNK FINALIZED [{current_time}] ═══╗\n"
            f"║ Duration: {duration:.1f}s\n"
            f"║ Samples: {len(phrase_array)}\n"
            f"╚{'═' * 40}╝"
        )

        # Отправляем в очередь обработки
        await self.batch_queue.add_batch(phrase_array)

        finalize_time = time.time() - finalize_start
        self.logger.debug(f"  → Finalize took {finalize_time*1000:.1f}ms")
