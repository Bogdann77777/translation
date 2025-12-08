"""
Модуль для работы с Groq Whisper API (Speech-to-Text).
"""

import asyncio
from groq import Groq
from typing import Dict, Any
from app.config import load_config, get_api_key
from app.monitoring.logger import setup_logger
from app.components.audio_utils import audio_to_wav_bytes


class GroqWhisperClient:
    """
    Клиент для Groq Whisper API.
    
    Функционал:
        - Транскрибирует аудио через Groq API
        - Автоматический retry при ошибках
        - Поддержка timeout
    """
    
    def __init__(self):
        """Инициализация клиента."""
        self.config = load_config()["models"]["whisper"]
        self.logger = setup_logger(__name__)
        
        # API клиент
        api_key = get_api_key(self.config["api_key_env"])
        self.client = Groq(api_key=api_key)
        
        # Параметры
        self.model = self.config["model"]
        self.language = self.config["language"]
        self.temperature = self.config["temperature"]
        
        self.logger.info(f"Groq Whisper client initialized: {self.model}")
    
    async def transcribe(self, audio_array) -> Dict[str, Any]:
        """
        Транскрибирует аудио через Groq Whisper API с retry логикой.
        
        Args:
            audio_array: Numpy массив с аудио (float32, 16kHz)
        
        Returns:
            dict: {"text": "...", "language": "en"}
        
        Алгоритм:
            1. Конвертируем numpy → WAV bytes
            2. Пытаемся вызвать API с exponential backoff
            3. При ошибке повторяем до max_attempts
            4. Возвращаем результат или raise
        """
        # Конвертируем в WAV
        wav_bytes = audio_to_wav_bytes(audio_array, 16000)
        
        # Retry параметры из конфига
        retry_config = load_config()["pipeline"]["retry"]
        max_attempts = retry_config["max_attempts"]
        backoff_factor = retry_config["backoff_factor"]
        
        # Пытаемся с retry
        for attempt in range(max_attempts):
            try:
                # Создаём file-like объект (каждый раз заново)
                from io import BytesIO
                audio_file = BytesIO(wav_bytes)
                audio_file.name = "audio.wav"
                
                # Вызываем API
                response = await asyncio.to_thread(
                    self.client.audio.transcriptions.create,
                    model=self.model,
                    file=audio_file,
                    language=self.language,
                    temperature=self.temperature
                )
                
                result = {
                    "text": response.text.strip(),
                    "language": self.language
                }
                
                self.logger.info(f"Transcribed: {len(result['text'])} chars")
                return result
                
            except Exception as e:
                wait_time = backoff_factor ** attempt
                self.logger.warning(
                    f"Groq API error (attempt {attempt + 1}/{max_attempts}): {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                
                if attempt < max_attempts - 1:
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Groq API failed after {max_attempts} attempts")
                    raise
