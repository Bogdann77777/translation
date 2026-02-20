"""
Модуль для работы с CosyVoice3 (Text-to-Speech).
Zero-shot voice cloning with high quality.
"""

import sys
import os
import torch
import numpy as np
import asyncio
from typing import Optional
from app.config import load_config
from app.monitoring.logger import setup_logger
from app.components.audio_utils import audio_to_wav_bytes, resample_audio


class CosyVoiceEngine:
    """
    Движок синтеза речи на основе CosyVoice3 (Alibaba).

    Функционал:
        - Zero-shot voice cloning (использует voice samples)
        - Поддержка русского языка
        - GPU ускорение
        - Латентность ~150ms
    """

    def __init__(self):
        """Инициализация CosyVoice движка."""
        self.config = load_config()["models"]["tts"]
        self.logger = setup_logger(__name__)

        try:
            # Параметры
            self.device = self.config.get("device", "cuda")
            self.gpu_id = self.config.get("gpu_id", 1)  # GPU 1 по умолчанию для TTS
            self.model_dir = self.config.get("model_dir", "models/CosyVoice3-0.5B")

            self.logger.info(f"Starting CosyVoice3 initialization: device={self.device}, gpu_id={self.gpu_id}, model_dir={self.model_dir}")

            # Voice sample - REQUIRED for zero-shot cloning
            self.voice_sample = self.config["voice_sample"]
            if self.voice_sample is None:
                raise ValueError("voice_sample is required for CosyVoice3! Please set it in config.yaml")

            # System prompt для zero-shot (можно настроить для разных стилей)
            self.system_prompt = self.config.get(
                "system_prompt",
                "You are a helpful assistant.<|endofprompt|>Говорите естественно и выразительно."
            )

            # Output sample rate (по умолчанию CosyVoice выдает 22050Hz)
            self.output_sample_rate = self.config.get("output_sample_rate", 24000)

            # Speech speed (1.0 = normal, 1.3 = 30% faster)
            # Note: CosyVoice supports speed control via instruct2 mode
            self.speed = self.config.get("speed", 1.0)

            # Language
            self.language = self.config.get("language", "ru")

            # Добавляем CosyVoice в Python path
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            cosyvoice_path = os.path.join(project_root, "third_party", "CosyVoice")
            matcha_path = os.path.join(cosyvoice_path, "third_party", "Matcha-TTS")

            if cosyvoice_path not in sys.path:
                sys.path.append(cosyvoice_path)
            if matcha_path not in sys.path:
                sys.path.append(matcha_path)

            self.logger.info(f"Added to Python path: {cosyvoice_path}")

            # Импортируем CosyVoice
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2 as CosyVoice
                self.logger.info("Using CosyVoice2 class (Fun-CosyVoice3-0.5B)")
            except ImportError:
                # Fallback to CosyVoice if CosyVoice2 not available
                from cosyvoice.cli.cosyvoice import CosyVoice
                self.logger.info("Using CosyVoice class (older version)")

            # Загружаем модель
            self.logger.info(f"Loading CosyVoice3 model from: {self.model_dir}")

            # Set device before loading
            if self.device == "cuda":
                torch.cuda.set_device(self.gpu_id)
                target_device = f"cuda:{self.gpu_id}"
            else:
                target_device = "cpu"

            # Load model
            import time
            t0 = time.time()
            self.model = CosyVoice(self.model_dir)
            t1 = time.time()

            # Move model to specific GPU if needed
            if hasattr(self.model, 'to'):
                self.model.to(target_device)

            self.logger.info(f"CosyVoice3 loaded in {t1-t0:.1f}s on {target_device}")

            # Get model's native sample rate
            self.model_sample_rate = getattr(self.model, 'sample_rate', 22050)

            self.logger.info(
                f"CosyVoice3 engine initialized: "
                f"voice={self.voice_sample}, lang={self.language}, "
                f"output={self.output_sample_rate}Hz, speed={self.speed}x"
            )

        except Exception as e:
            self.logger.error(f"CosyVoice3 initialization failed: {type(e).__name__}: {e}", exc_info=True)
            raise

    async def synthesize(self, text: str) -> bytes:
        """
        Синтезирует речь из текста с retry логикой (zero-shot voice cloning).

        Args:
            text: Текст для озвучки (на русском)

        Returns:
            bytes: WAV audio в бинарном формате

        Алгоритм:
            1. Вызываем CosyVoice inference_zero_shot с voice sample
            2. При ошибке повторяем с exponential backoff
            3. Получаем audio tensor
            4. Конвертируем в numpy и WAV bytes
            5. Возвращаем результат
        """
        self.logger.info(f"Starting synthesis: {len(text)} chars, first 100 chars: {text[:100]}")

        # Retry параметры из конфига
        retry_config = load_config()["pipeline"]["retry"]
        max_attempts = retry_config["max_attempts"]
        backoff_factor = retry_config["backoff_factor"]

        # Пытаемся с retry
        for attempt in range(max_attempts):
            try:
                self.logger.debug(f"Synthesis attempt {attempt + 1}/{max_attempts}")

                # Нормализуем текст в UTF-8
                if isinstance(text, str):
                    text_normalized = text.encode('utf-8', errors='ignore').decode('utf-8')
                    self.logger.debug(f"Text normalized: {len(text_normalized)} chars")
                else:
                    text_normalized = text

                # CosyVoice inference_zero_shot
                self.logger.debug(
                    f"Calling CosyVoice inference_zero_shot: "
                    f"voice_sample={self.voice_sample}, prompt={self.system_prompt[:50]}..."
                )

                # Синтез через CosyVoice (в отдельном потоке т.к. синхронная операция)
                output = await asyncio.to_thread(
                    self._synthesize_sync,
                    text_normalized
                )

                # Получаем audio из результата
                # output - это generator, берем первый результат
                audio_tensor = output['tts_speech']

                self.logger.debug(f"CosyVoice returned audio tensor: shape={audio_tensor.shape}")

                # Конвертируем tensor в numpy
                if isinstance(audio_tensor, torch.Tensor):
                    audio_array = audio_tensor.cpu().numpy()
                else:
                    audio_array = np.array(audio_tensor, dtype=np.float32)

                # Убираем лишние измерения если есть
                if audio_array.ndim > 1:
                    audio_array = audio_array.squeeze()

                # Resample если нужно (CosyVoice обычно 22050Hz, нам нужно 24000Hz)
                if self.model_sample_rate != self.output_sample_rate:
                    audio_array = resample_audio(
                        audio_array,
                        self.model_sample_rate,
                        self.output_sample_rate
                    )
                    self.logger.debug(f"Resampled: {self.model_sample_rate}Hz → {self.output_sample_rate}Hz")

                # Применяем speed если нужно (через изменение sample rate)
                if self.speed != 1.0:
                    # Speed up/down через resampling
                    # speed=1.3 означает 30% быстрее = уменьшить sample rate при воспроизведении
                    # Но для WAV файла нужно изменить данные
                    adjusted_rate = int(self.output_sample_rate / self.speed)
                    audio_array = resample_audio(audio_array, self.output_sample_rate, adjusted_rate)
                    audio_array = resample_audio(audio_array, adjusted_rate, self.output_sample_rate)
                    self.logger.debug(f"Applied speed adjustment: {self.speed}x")

                # Конвертируем в WAV bytes
                wav_bytes = audio_to_wav_bytes(audio_array, self.output_sample_rate)

                self.logger.info(
                    f"Synthesized: {len(text)} chars → {len(wav_bytes)} bytes "
                    f"({self.output_sample_rate}Hz, {self.speed}x speed)"
                )
                return wav_bytes

            except Exception as e:
                wait_time = backoff_factor ** attempt
                self.logger.warning(
                    f"CosyVoice synthesis error (attempt {attempt + 1}/{max_attempts}): {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )

                if attempt < max_attempts - 1:
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"CosyVoice synthesis failed after {max_attempts} attempts")
                    raise

    def _synthesize_sync(self, text: str) -> dict:
        """
        Синхронная обертка для CosyVoice inference.

        Args:
            text: Текст для синтеза

        Returns:
            dict: Результат с 'tts_speech' tensor
        """
        # inference_zero_shot возвращает generator
        # Берем первый результат
        for i, output in enumerate(self.model.inference_zero_shot(
            text,
            self.system_prompt,
            self.voice_sample,
            stream=False  # Не используем streaming для простоты
        )):
            if i == 0:
                return output

        raise RuntimeError("CosyVoice inference_zero_shot returned empty generator")
