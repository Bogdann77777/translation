"""
Модуль для работы с XTTS-v2 (Text-to-Speech).
"""

import torch
import numpy as np

# CRITICAL FIX 1: PyTorch 2.6+ changed default weights_only=True, breaking TTS model loading
# Solution: Patch torch.load to use weights_only=False by default
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    """Patch for torch.load with weights_only=False for TTS model loading."""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# CRITICAL FIX 1b: Removed patched_load_fsspec because TTS 0.22.0+ already passes weights_only=False
# import TTS.utils.io
# _original_load_fsspec = TTS.utils.io.load_fsspec
# def _patched_load_fsspec(path, map_location=None, **kwargs):
#     """Patch for TTS load_fsspec with weights_only=False."""
#     if 'weights_only' not in kwargs:
#         kwargs['weights_only'] = False
#     return _original_load_fsspec(path, map_location=map_location, **kwargs)
# TTS.utils.io.load_fsspec = _patched_load_fsspec

# CRITICAL FIX 2: Patch torchaudio.load to use soundfile instead of torchcodec
# Reason: torchcodec requires FFmpeg which may not be available on Windows
import torchaudio
_original_torchaudio_load = torchaudio.load

def _patched_torchaudio_load(filepath, *args, **kwargs):
    """Патч для torchaudio.load - использует soundfile вместо torchcodec."""
    import soundfile as sf
    audio, sr = sf.read(filepath)
    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio).float()
    # Add channel dimension if mono
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    elif audio_tensor.ndim == 2:
        # soundfile returns (samples, channels), torchaudio expects (channels, samples)
        audio_tensor = audio_tensor.transpose(0, 1)
    return audio_tensor, sr

torchaudio.load = _patched_torchaudio_load

# CRITICAL FIX 3: Patch GPT2InferenceModel for transformers >= 4.50
# Issue: Newer transformers removed GenerationMixin from PreTrainedModel, breaking TTS generate() call
try:
    from transformers.generation import GenerationMixin
    from TTS.tts.layers.xtts.gpt_inference import GPT2InferenceModel
    
    if GenerationMixin not in GPT2InferenceModel.__bases__:
        # Inject GenerationMixin into MRO
        GPT2InferenceModel.__bases__ = (GenerationMixin,) + GPT2InferenceModel.__bases__
except Exception as e:
    print(f"Warning: Failed to patch GPT2InferenceModel: {e}")

from TTS.api import TTS
from typing import Optional
from app.config import load_config
from app.monitoring.logger import setup_logger
from app.components.audio_utils import float32_to_int16, audio_to_wav_bytes, resample_audio


class XTTSEngine:
    """
    Движок синтеза речи на основе XTTS-v2.
    
    Функционал:
        - Синтез русской речи из текста
        - Использует voice sample для клонирования
        - GPU ускорение
    """
    
    def __init__(self):
        """Инициализация TTS движка."""
        self.config = load_config()["models"]["tts"]
        self.logger = setup_logger(__name__)

        try:
            # Загружаем модель
            self.device = self.config["device"]
            self.gpu_id = self.config.get("gpu_id", 1)  # GPU 1 по умолчанию для TTS

            self.logger.info(f"Starting XTTS initialization: device={self.device}, gpu_id={self.gpu_id}")

            # REMOVED: torch.cuda.set_device(self.gpu_id) - causes conflicts with faster-whisper
            # self.logger.info(f"Setting XTTS to GPU {self.gpu_id}")

            self.logger.info(f"Loading TTS model: {self.config['model']}")

            import time
            t0 = time.time()
            self.logger.info("[XTTS] Step 1: Creating TTS object...")

            # Initialize on CPU first, then move to specific GPU
            self.model = TTS(
                model_name=self.config["model"],
                gpu=False 
            )
            
            # Move to correct device
            target_device = f"{self.device}:{self.gpu_id}" if self.device == "cuda" else "cpu"
            self.model.to(target_device)

            t1 = time.time()
            self.logger.info(f"[XTTS] Step 1 done in {t1-t0:.1f}s - TTS object created")
            self.logger.info("TTS model loaded successfully")

            # Voice sample - REQUIRED for XTTS v2 (multi-speaker model)
            # Note: torchaudio.load is patched above to use soundfile instead of torchcodec
            self.voice_sample = self.config["voice_sample"]
            if self.voice_sample is None:
                raise ValueError("voice_sample is required for XTTS v2! Please set it in config.yaml")

            self.language = self.config["language"]
            self.logger.info(f"Voice sample: {self.voice_sample}, language: {self.language}")

            # Output sample rate (24kHz default, 48kHz for browser compatibility)
            self.output_sample_rate = self.config.get("output_sample_rate", 24000)

            # Speech speed (1.0 = normal, 1.3 = 30% faster)
            self.speed = self.config.get("speed", 1.0)

            self.logger.info(f"XTTS engine initialized on {self.device}:{self.gpu_id} (output: {self.output_sample_rate}Hz, speed: {self.speed}x)")

        except Exception as e:
            self.logger.error(f"XTTS initialization failed: {type(e).__name__}: {e}", exc_info=True)
            raise
    
    async def synthesize(self, text: str) -> bytes:
        """
        Синтезирует русскую речь из текста с retry логикой.
        
        Args:
            text: Русский текст для озвучки
        
        Returns:
            bytes: WAV audio в бинарном формате (24kHz)
        
        Алгоритм:
            1. Вызываем TTS модель с voice cloning
            2. При ошибке повторяем с exponential backoff
            3. Получаем numpy массив
            4. Конвертируем в WAV bytes
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

                # Нормализуем текст в UTF-8 (fix Windows charmap encoding issue)
                if isinstance(text, str):
                    # Убеждаемся что текст корректно обрабатывается как UTF-8
                    text_normalized = text.encode('utf-8', errors='ignore').decode('utf-8')
                    self.logger.debug(f"Text normalized: {len(text_normalized)} chars")
                else:
                    text_normalized = text

                # Синтез через XTTS
                import asyncio
                self.logger.debug(f"Calling TTS with voice_sample={self.voice_sample}, language={self.language}")

                # Build TTS arguments
                tts_kwargs = {
                    "text": text_normalized,
                    "language": self.language,
                    "speed": self.speed  # Speed multiplier (1.0 = normal)
                }

                # Only add speaker_wav if voice_sample is not None
                if self.voice_sample is not None:
                    tts_kwargs["speaker_wav"] = self.voice_sample

                audio_array = await asyncio.to_thread(
                    self.model.tts,
                    **tts_kwargs
                )
                self.logger.debug(f"TTS returned audio array: shape={np.array(audio_array).shape if hasattr(audio_array, '__len__') else 'scalar'}")
                
                # Конвертируем в numpy если нужно
                if not isinstance(audio_array, np.ndarray):
                    audio_array = np.array(audio_array, dtype=np.float32)

                # Resample if output_sample_rate != 24000 (native XTTS rate)
                if self.output_sample_rate != 24000:
                    audio_array = resample_audio(audio_array, 24000, self.output_sample_rate)
                    self.logger.debug(f"Resampled: 24kHz → {self.output_sample_rate}Hz")

                # Конвертируем в WAV bytes
                wav_bytes = audio_to_wav_bytes(audio_array, self.output_sample_rate)

                self.logger.info(f"Synthesized: {len(text)} chars → {len(wav_bytes)} bytes ({self.output_sample_rate}Hz)")
                return wav_bytes
                
            except Exception as e:
                wait_time = backoff_factor ** attempt
                self.logger.warning(
                    f"XTTS synthesis error (attempt {attempt + 1}/{max_attempts}): {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                
                if attempt < max_attempts - 1:
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"XTTS synthesis failed after {max_attempts} attempts")
                    raise
