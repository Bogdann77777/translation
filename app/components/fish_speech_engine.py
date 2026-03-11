"""
Fish Speech 1.5 TTS engine — HTTP-клиент к fish-speech API серверу.
Интерфейс идентичен TTSWorkerPool: synthesize(text) -> bytes, set_speed(speed), shutdown().
Без multiprocessing — GPU уже в отдельном процессе сервера.
"""

import asyncio
import struct
import requests
from app.config import load_config
from app.monitoring.logger import setup_logger


class FishSpeechTTSPool:
    """
    TTS pool для fish-speech 1.5 API.

    Отличие от TTSWorkerPool:
    - Нет multiprocessing (сервер — отдельный процесс)
    - Скорость применяется через atempo в batch_queue.py (адаптивно по очереди)
    - current_speed хранит базовую скорость от слайдера
    """

    def __init__(self):
        cfg = load_config()["models"]["tts"]
        self.logger = setup_logger(__name__)

        self.api_url = cfg.get("fish_speech_url", "http://localhost:8080")
        self.voice_sample = cfg["voice_sample"]
        self.temperature = cfg.get("temperature", 0.7)
        self.top_p = cfg.get("top_p", 0.8)
        self.repetition_penalty = cfg.get("repetition_penalty", 1.1)
        self.current_speed = 1.0  # базовая скорость от слайдера
        self.output_sample_rate = 44100  # fish-speech native rate

        # Загружаем референсное аудио один раз
        with open(self.voice_sample, "rb") as f:
            self._ref_audio = f.read()

        self.logger.info(
            f"FishSpeechTTSPool ready: api={self.api_url}, "
            f"voice={self.voice_sample}, temp={self.temperature}"
        )

    async def synthesize(self, text: str) -> bytes:
        """
        Синтезирует речь через fish-speech API.
        Возвращает WAV байты (44100 Hz).
        """
        return await asyncio.to_thread(self._call_api, text)

    def _call_api(self, text: str) -> bytes:
        """Синхронный HTTP вызов к fish-speech API."""
        import ormsgpack

        payload = ormsgpack.packb({
            "text": text,
            "references": [{"audio": self._ref_audio, "text": ""}],
            "format": "wav",
            "streaming": False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "chunk_length": 200,
            "max_new_tokens": 1024,
            "seed": None,
            "use_memory_cache": "on",
        })

        r = requests.post(
            f"{self.api_url}/v1/tts",
            data=payload,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "audio/wav",
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.content

    def set_speed(self, speed: float) -> None:
        """Устанавливает базовую скорость (от слайдера). Применяется через atempo при воспроизведении."""
        self.current_speed = speed
        self.logger.info(f"FishSpeech base speed set to {speed}x")

    async def shutdown(self) -> None:
        """Нет ресурсов для освобождения."""
        pass
