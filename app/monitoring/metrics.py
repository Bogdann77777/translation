"""
Модуль для сбора метрик производительности системы.
Отслеживает latency, VRAM, ошибки и throughput.
"""

import time
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any
from app.config import load_config
from app.monitoring.logger import setup_logger


class MetricsCollector:
    """
    Собирает метрики производительности системы.
    
    Отслеживаемые метрики:
        - Latency (STT, LLM, TTS, End-to-End)
        - VRAM usage (GPU memory)
        - Error counts (по стадиям)
        - Batch throughput (батчей/сек)
    
    Example:
        >>> metrics = MetricsCollector()
        >>> metrics.record_latency("stt", 0.3)
        >>> metrics.record_latency("translation", 1.5)
        >>> summary = metrics.get_summary()
        >>> print(summary["latency_avg"])
        {'stt': 0.3, 'translation': 1.5, ...}
    """
    
    def __init__(self):
        """
        Инициализация коллектора метрик.
        
        Атрибуты:
            config: dict - Конфигурация из config.yaml
            logger: logging.Logger - Логгер для записи метрик
            session_start: float - Время старта сессии (time.time())
            batches_processed: int - Счётчик обработанных батчей
            total_audio_seconds: float - Общее время аудио (сек)
            latencies: defaultdict(list) - Словарь списков задержек
            errors: defaultdict(int) - Счётчик ошибок
            vram_usage: list - История использования VRAM
        
        Example:
            >>> metrics = MetricsCollector()
            >>> print(metrics.session_start)
            1698765432.123
        """
        self.config = load_config()
        self.logger = setup_logger(__name__)
        
        self.session_start = time.time()
        self.batches_processed = 0
        self.total_audio_seconds = 0.0
        
        self.latencies = defaultdict(list)
        self.errors = defaultdict(int)
        self.vram_usage = []
    
    def record_latency(self, stage: str, duration: float) -> None:
        """
        Записывает задержку для определённой стадии pipeline.
        
        Stages:
            - "stt": Speech-to-Text (Groq Whisper)
            - "translation": Translation (OpenRouter LLM)
            - "tts": Text-to-Speech (XTTS-v2)
            - "e2e": End-to-End (full pipeline)
        
        Алгоритм:
            1. Добавляем duration в список self.latencies[stage]
            2. Получаем threshold из конфига
            3. Если duration > threshold → логируем warning
            4. Это помогает отслеживать аномально медленные операции
        
        Args:
            stage: Название стадии ("stt", "translation", "tts", "e2e")
            duration: Время выполнения в секундах
        
        Example:
            >>> metrics = MetricsCollector()
            >>> metrics.record_latency("stt", 0.3)
            >>> metrics.record_latency("translation", 1.5)
            >>> metrics.record_latency("tts", 2.1)
        """
        # Добавляем в список
        self.latencies[stage].append(duration)
        
        # Проверяем threshold
        threshold = self.config["monitoring"]["metrics"]["latency_alert_threshold"]
        if duration > threshold:
            self.logger.warning(
                f"High latency in {stage}: {duration:.2f}s "
                f"(threshold: {threshold}s)"
            )
    
    def record_error(self, stage: str, error_type: str) -> None:
        """
        Записывает ошибку для определённой стадии.
        
        Алгоритм:
            1. Увеличиваем счётчик self.errors[stage] += 1
            2. Логируем ошибку через logger.error()
            3. Это помогает отслеживать проблемные компоненты
        
        Args:
            stage: Название стадии ("stt", "translation", "tts")
            error_type: Тип ошибки ("timeout", "api_error", "network", etc.)
        
        Example:
            >>> metrics.record_error("stt", "groq_api_timeout")
            >>> metrics.record_error("translation", "openrouter_rate_limit")
        """
        # Увеличиваем счётчик
        self.errors[stage] += 1
        
        # Логируем ошибку
        self.logger.error(
            f"Error in {stage}: {error_type} "
            f"(total errors in {stage}: {self.errors[stage]})"
        )
    
    def get_vram_usage(self) -> float:
        """
        Получает текущее использование VRAM (GPU memory).
        
        Алгоритм:
            1. Проверяем, доступна ли CUDA через torch.cuda.is_available()
            2. Если нет → возвращаем 0.0 (CPU mode)
            3. Если да:
               - Получаем байты через torch.cuda.memory_allocated()
               - Конвертируем в MB: bytes / (1024**2)
               - Добавляем значение в историю self.vram_usage
            4. Проверяем threshold
            5. Если превышен → логируем warning
            6. Возвращаем текущее значение
        
        Returns:
            float: VRAM в мегабайтах (MB)
        
        Example:
            >>> vram = metrics.get_vram_usage()
            >>> print(f"VRAM: {vram:.1f} MB")
            VRAM: 3245.2 MB
        """
        # Проверяем CUDA
        if not torch.cuda.is_available():
            return 0.0
        
        # Получаем VRAM в MB
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        self.vram_usage.append(vram_mb)
        
        # Проверяем threshold
        threshold = self.config["monitoring"]["metrics"]["vram_alert_threshold"]
        if vram_mb > threshold:
            self.logger.warning(
                f"High VRAM usage: {vram_mb:.1f} MB "
                f"(threshold: {threshold} MB)"
            )
        
        return vram_mb
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку всех метрик за сессию.
        
        Алгоритм:
            1. Вычисляем session_duration = time.time() - session_start
            2. Для каждой стадии считаем среднюю latency через np.mean()
            3. Получаем текущий VRAM через get_vram_usage()
            4. Вычисляем средний VRAM: np.mean(self.vram_usage)
            5. Вычисляем пиковый VRAM: max(self.vram_usage)
            6. Возвращаем всё в dict
        
        Returns:
            Dict[str, Any]: Структура с агрегированными метриками:
                {
                    "session_duration": 3600.5,
                    "batches_processed": 245,
                    "total_audio_seconds": 1837.2,
                    "latency_avg": {
                        "stt": 0.32,
                        "translation": 1.48,
                        "tts": 2.03,
                        "e2e": 3.83
                    },
                    "errors": {
                        "stt": 0,
                        "translation": 2,
                        "tts": 1
                    },
                    "vram_current_mb": 3245.2,
                    "vram_avg_mb": 3102.7,
                    "vram_peak_mb": 3890.1
                }
        """
        session_duration = time.time() - self.session_start
        
        latency_avg = {
            stage: np.mean(times) if times else 0.0
            for stage, times in self.latencies.items()
        }
        
        vram_current = self.get_vram_usage()
        vram_avg = np.mean(self.vram_usage) if self.vram_usage else 0.0
        vram_peak = max(self.vram_usage) if self.vram_usage else 0.0
        
        return {
            "session_duration": session_duration,
            "batches_processed": self.batches_processed,
            "total_audio_seconds": self.total_audio_seconds,
            "latency_avg": latency_avg,
            "errors": dict(self.errors),
            "vram_current_mb": vram_current,
            "vram_avg_mb": vram_avg,
            "vram_peak_mb": vram_peak
        }
