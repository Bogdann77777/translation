"""
Модуль для загрузки и валидации конфигурации.
Обеспечивает единую точку доступа к настройкам проекта.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Пути к файлам конфигурации
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
ENV_PATH = Path(__file__).parent.parent / ".env"

# Кэш конфигурации (загружается 1 раз)
_config_cache: Optional[Dict[str, Any]] = None


def load_config() -> Dict[str, Any]:
    """
    Загружает конфигурацию из config.yaml с кэшированием.
    
    Алгоритм:
        1. Проверяем кэш (_config_cache)
        2. Если есть → возвращаем (быстро)
        3. Если нет:
           - Проверяем существование config.yaml
           - Парсим YAML
           - Валидируем структуру
           - Сохраняем в кэш
           - Возвращаем результат
    
    Returns:
        Dict[str, Any]: Полная конфигурация системы
    
    Raises:
        FileNotFoundError: Если config.yaml не найден
        yaml.YAMLError: Если файл повреждён
    
    Example:
        >>> config = load_config()
        >>> whisper_model = config["models"]["whisper"]["model"]
        >>> print(whisper_model)
        'whisper-large-v3'
    """
    global _config_cache
    
    # Проверяем кэш
    if _config_cache is not None:
        return _config_cache
    
    # Проверяем существование файла
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found: {CONFIG_PATH}\n"
            f"Please create config.yaml in project root."
        )
    
    # Читаем и парсим YAML
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Failed to parse config.yaml: {e}\n"
            f"Check YAML syntax."
        )
    
    # Валидируем структуру
    validate_config(config)
    
    # Сохраняем в кэш
    _config_cache = config
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Проверяет наличие обязательных ключей в конфигурации.
    
    Алгоритм:
        1. Создаём список обязательных ключей (кортежи путей)
        2. Для каждого ключа проходим по цепочке dict
        3. Если хоть один отсутствует → ValueError
        4. Если все на месте → return (успех)
    
    Args:
        config: Словарь конфигурации
    
    Raises:
        ValueError: Если отсутствуют критичные ключи
    
    Example:
        >>> config = {"models": {"whisper": {}}}
        >>> validate_config(config)  # Raises ValueError
    """
    # Список обязательных ключей
    required_keys = [
        ("models", "whisper"),
        ("models", "translation"),
        ("models", "tts"),
        ("pipeline", "context_window"),
        ("pipeline", "batch_queue_size"),
        ("pipeline", "vad"),
        ("pipeline", "audio"),
        ("server", "host"),
        ("server", "port"),
        ("monitoring", "enabled"),
    ]
    
    # Проверяем каждый ключ
    for keys in required_keys:
        temp = config
        for key in keys:
            if key not in temp:
                raise ValueError(
                    f"Missing required config key: {'.'.join(keys)}\n"
                    f"Please check config.yaml structure."
                )
            temp = temp[key]
    
    # Все ключи на месте
    print("[OK] Config validation passed")


def get_api_key(key_name: str) -> str:
    """
    Получает API ключ из .env файла.
    
    Алгоритм:
        1. Загружаем .env файл через load_dotenv()
        2. Читаем переменную через os.getenv(key_name)
        3. Если None → ValueError
        4. Если есть → возвращаем строку
    
    Args:
        key_name: Имя переменной окружения (например, "GROQ_API_KEY")
    
    Returns:
        str: Значение API ключа
    
    Raises:
        ValueError: Если ключ не найден в .env
    
    Security:
        - Ключи НЕ логируются
        - .env файл в .gitignore
        - Никогда не выводим ключи в консоль
    
    Example:
        >>> groq_key = get_api_key("GROQ_API_KEY")
        >>> print(groq_key[:10])  # Показываем только первые 10 символов
        'gsk_abc123'
    """
    # Загружаем .env файл
    load_dotenv(ENV_PATH)
    
    # Получаем ключ
    api_key = os.getenv(key_name)
    
    # Проверяем наличие
    if api_key is None or api_key == "":
        raise ValueError(
            f"API key not found: {key_name}\n"
            f"Please add {key_name}=your_key_here to .env file.\n"
            f"Expected file location: {ENV_PATH}"
        )
    
    return api_key
