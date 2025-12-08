"""
Модуль детекции речи (Voice Activity Detection).
Использует Silero VAD v5 для определения наличия речи в аудио.
"""

import torch
import numpy as np
from typing import Optional
from app.config import load_config
from app.monitoring.logger import setup_logger


class VADDetector:
    """
    Детектор речи на основе Silero VAD v5.
    
    Функционал:
        - Определяет наличие речи в аудио чанке
        - Определяет тишину для финализации фраз
        - Использует порог (threshold) для чувствительности
    
    Параметры из config.yaml:
        - vad.enabled: bool
        - vad.threshold: float (0.0-1.0)
        - vad.min_speech_duration: float (секунды)
        - vad.min_silence_duration: float (секунды)
    """
    
    def __init__(self):
        """
        Инициализация VAD детектора.
        """
        self.config = load_config()["pipeline"]["vad"]
        self.logger = setup_logger(__name__)
        
        # Загружаем Silero VAD v5
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        # ВАЖНО: Silero VAD v5 не поддерживает SM 12.0 (RTX 5060 Ti)
        # Используем CPU для VAD (быстро, <1ms на чанк)
        self.device = torch.device('cpu')
        self.model.to(self.device)
        
        # Параметры
        self.threshold = self.config["threshold"]
        self.min_speech_duration = self.config["min_speech_duration"]
        self.min_silence_duration = self.config["min_silence_duration"]
        
        # Счётчики фреймов
        self.speech_frames = 0
        self.silence_frames = 0
        
        self.logger.info(f"VAD initialized on {self.device}")
    
    def detect_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Определяет наличие речи в аудио чанке.
        
        Args:
            audio_chunk: Аудио массив (float32, 16kHz)
        
        Returns:
            bool: True если речь обнаружена, False если тишина
        
        Note:
            Silero VAD v5 требует строго 512 samples для 16kHz.
            Разбиваем большие чанки на окна по 512 samples.
        """
        # Silero VAD v5 требует ровно 512 samples
        window_size = 512
        
        # Если чанк меньше окна - дополняем нулями
        if len(audio_chunk) < window_size:
            audio_chunk = np.pad(audio_chunk, (0, window_size - len(audio_chunk)))
        
        # Обрабатываем по окнам 512 samples
        speech_probs = []
        for i in range(0, len(audio_chunk), window_size):
            window = audio_chunk[i:i+window_size]
            
            # Дополняем последнее окно если нужно
            if len(window) < window_size:
                window = np.pad(window, (0, window_size - len(window)))
            
            # VAD для этого окна
            audio_tensor = torch.from_numpy(window).to(self.device)
            with torch.no_grad():
                prob = self.model(audio_tensor, 16000).item()
            speech_probs.append(prob)
        
        # Усредняем вероятность по всем окнам
        avg_prob = np.mean(speech_probs)
        
        # Проверяем threshold
        if avg_prob > self.threshold:
            self.speech_frames += 1
            self.silence_frames = 0
            return True
        else:
            self.silence_frames += 1
            self.speech_frames = 0
            return False
    
    def is_silence_ready(self) -> bool:
        """
        Проверяет, достаточно ли накопилось тишины для финализации фразы.

        ВАЖНО: Не вызывает detect_speech()! Использует уже накопленный
        silence_frames счётчик. Вызывать ПОСЛЕ detect_speech().

        Returns:
            bool: True если достаточно тишины для финализации

        Note:
            min_silence_duration из конфига определяет порог.
            При 100ms чанках: 1.5s = 15 фреймов тишины.
        """
        min_silence_frames = int(self.min_silence_duration * 10)
        return self.silence_frames >= min_silence_frames

    def reset(self) -> None:
        """
        Сбрасывает счётчики после финализации фразы.
        Вызывать после каждого finalize_phrase().
        """
        self.speech_frames = 0
        self.silence_frames = 0
