"""
Модуль для настройки структурированных логов.
Поддерживает JSON логирование в файл и plain text в консоль.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from app.config import load_config

# Кэш логгеров (singleton pattern)
_loggers: Dict[str, logging.Logger] = {}


def setup_logger(name: str) -> logging.Logger:
    """
    Создаёт или возвращает существующий logger с настройками.
    
    Настройки:
        - File handler: JSON формат для парсинга (Grafana/ELK/Loki)
        - Console handler: Plain text для human reading
        - Уровень логирования: из config.yaml (INFO/DEBUG/WARNING/ERROR)
        - Автоматическое создание директории логов
    
    Алгоритм:
        1. Проверяем кэш (_loggers)
        2. Если логгер уже создан → возвращаем его (singleton pattern)
        3. Если нет:
           - Загружаем конфиг (log_level, log_dir)
           - Создаём logger через logging.getLogger(name)
           - Устанавливаем уровень (INFO/DEBUG/WARNING/ERROR)
           - Создаём директорию для логов (если не существует)
           - Создаём file handler → logs/session_YYYY-MM-DD.log
           - Создаём console handler → stdout
           - Добавляем JSONFormatter к file handler
           - Добавляем обычный Formatter к console handler
           - Добавляем handlers к logger
           - Сохраняем в кэш _loggers[name]
           - Возвращаем logger
    
    Args:
        name: Имя логгера (обычно __name__ модуля)
    
    Returns:
        logging.Logger: Настроенный логгер с handlers
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Starting translation pipeline")
    """
    # Проверяем кэш
    if name in _loggers:
        return _loggers[name]
    
    # Загружаем конфиг
    config = load_config()
    log_level = config["monitoring"]["log_level"]
    log_dir = Path(config["monitoring"]["log_dir"])
    
    # Создаём директорию
    log_dir.mkdir(exist_ok=True)
    
    # Создаём logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    # File handler (JSON для парсинга)
    log_file = log_dir / f"session_{datetime.now().strftime('%Y-%m-%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(JSONFormatter())
    
    # Console handler (plain text для human reading)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Добавляем handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Кэшируем
    _loggers[name] = logger
    
    return logger



class JSONFormatter(logging.Formatter):
    """
    Formatter для структурированных JSON логов.
    
    Output format:
        {
            "timestamp": "2025-10-30T15:30:45.123Z",
            "level": "INFO",
            "logger": "app.pipeline.orchestrator",
            "message": "Batch processed",
            "extra": {"batch_id": 123, "latency": 3.2}
        }
    
    Example:
        >>> formatter = JSONFormatter()
        >>> handler.setFormatter(formatter)
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Форматирует лог запись в JSON.
        
        Алгоритм:
            1. Создаём словарь log_data с полями:
               - timestamp: ISO формат UTC
               - level: INFO/DEBUG/WARNING/ERROR
               - logger: имя логгера
               - message: текст сообщения
            2. Если есть атрибут 'extra' → добавляем в log_data["extra"]
            3. Конвертируем в JSON строку через json.dumps()
            4. Возвращаем строку
        
        Args:
            record: logging.LogRecord объект
        
        Returns:
            str: JSON строка
        """
        # Базовые поля
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        
        # Добавляем extra поля если есть
        if hasattr(record, 'extra'):
            log_data["extra"] = record.extra
        
        # Конвертируем в JSON
        return json.dumps(log_data)



def log_json(logger: logging.Logger, level: str, message: str, **kwargs) -> None:
    """
    Удобная функция для логирования с дополнительными данными.
    
    Используется для structured logging - добавляет произвольные
    поля в JSON лог для последующего анализа.
    
    Алгоритм:
        1. Получаем метод логгера через getattr(logger, level.lower())
        2. Вызываем метод с message и extra={'extra': kwargs}
        3. JSONFormatter автоматически добавит kwargs в поле "extra"
    
    Args:
        logger: Logger объект
        level: Уровень лога ("INFO", "DEBUG", "WARNING", "ERROR")
        message: Текст сообщения
        **kwargs: Дополнительные поля для extra
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> log_json(logger, "INFO", "Batch processed",
        ...          batch_id=123, latency=3.2, stage="tts")
    
    Output (в logs/session_2025-10-30.log):
        {
            "timestamp": "2025-10-30T15:30:45.123Z",
            "level": "INFO",
            "logger": "app.pipeline.batch_queue",
            "message": "Batch processed",
            "extra": {
                "batch_id": 123,
                "latency": 3.2,
                "stage": "tts"
            }
        }
    """
    # Получаем метод логирования (info, debug, warning, error)
    log_method = getattr(logger, level.lower())
    
    # Логируем с extra данными
    log_method(message, extra={'extra': kwargs})
