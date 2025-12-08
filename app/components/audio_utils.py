"""
Модуль утилит для обработки аудио.
Содержит функции для ресемплинга, конверсии форматов и нормализации.
"""

import numpy as np
from scipy.signal import resample
from io import BytesIO
import wave
from typing import cast


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    Ресемплирует аудио из одной частоты в другую.
    
    Используется для:
        - Конверсия микрофонного входа (48kHz) → Whisper API (16kHz)
        - Конверсия XTTS output (24kHz) → Browser playback (48kHz)
    
    Алгоритм:
        1. Проверяем, нужен ли ресемплинг (src_rate == dst_rate?)
        2. Если частоты одинаковые → return audio (без изменений)
        3. Если разные:
           - Вычисляем новую длину: num_samples = len(audio) * dst_rate / src_rate
           - Используем scipy.signal.resample(audio, num_samples)
           - Возвращаем ресемплированный массив с исходным dtype
    
    Args:
        audio: Аудио массив (float32 или int16)
        src_rate: Исходная частота (например, 48000)
        dst_rate: Целевая частота (например, 16000)
    
    Returns:
        Ресемплированное аудио той же размерности
        
    Example:
        >>> # 48kHz → 16kHz (для Whisper API)
        >>> audio_48k = np.random.randn(48000).astype(np.float32)
        >>> audio_16k = resample_audio(audio_48k, 48000, 16000)
        >>> print(len(audio_16k))
        16000
    
    Note:
        - Используем scipy.signal.resample (не librosa!)
        - Работает с float32 и int16 автоматически
        - Сохраняет dtype исходного массива
    """
    # Проверяем, нужен ли ресемплинг
    if src_rate == dst_rate:
        return audio
    
    # Вычисляем новую длину
    num_samples = int(len(audio) * dst_rate / src_rate)

    # Ресемплируем через scipy
    resampled: np.ndarray = cast(np.ndarray, resample(audio, num_samples))

    # Сохраняем исходный dtype
    return resampled.astype(audio.dtype)



def float32_to_int16(audio_float32: np.ndarray) -> np.ndarray:
    """
    Конвертирует float32 аудио [-1.0, 1.0] в int16 [-32768, 32767].
    
    Используется для:
        - WebSocket отправка (int16 компактнее)
        - VAD детектор (работает с int16)
        - Некоторые TTS модели (требуют int16)
    
    Алгоритм:
        1. Умножаем на 32768 (максимальное значение int16)
        2. Clip значения в диапазон [-32768, 32767]
        3. Приводим к типу int16
    
    Args:
        audio_float32: Аудио массив float32, значения [-1.0, 1.0]
    
    Returns:
        Аудио массив int16
    
    Formula:
        int16_value = np.clip(float32_value * 32768, -32768, 32767).astype(np.int16)
    
    Example:
        >>> audio_float = np.array([0.5, -0.8, 1.0])
        >>> audio_int = float32_to_int16(audio_float)
        >>> print(audio_int)
        [16384 -26214 32767]
    """
    # Умножаем на 32768 и clip в диапазон int16
    audio_int16 = np.clip(audio_float32 * 32768, -32768, 32767).astype(np.int16)
    return audio_int16



def int16_to_float32(audio_int16: np.ndarray) -> np.ndarray:
    """
    Конвертирует int16 аудио [-32768, 32767] в float32 [-1.0, 1.0].
    
    Используется для:
        - Whisper API (ожидает float32)
        - Обработка сигналов (удобнее во float32)
        - Математические операции (точнее в float)
    
    Алгоритм:
        1. Приводим к типу float32
        2. Делим на 32768.0 для нормализации в диапазон [-1.0, 1.0]
    
    Args:
        audio_int16: Аудио массив int16
    
    Returns:
        Аудио массив float32
    
    Formula:
        float32_value = int16_value.astype(np.float32) / 32768.0
    
    Example:
        >>> audio_int = np.array([16384, -26214, 32767], dtype=np.int16)
        >>> audio_float = int16_to_float32(audio_int)
        >>> print(audio_float)
        [0.5 -0.8 0.99997]
    """
    # Конвертируем в float32 и нормализуем
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32



def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """
    Конвертирует numpy массив в WAV bytes для отправки по сети.
    
    Используется для:
        - Groq Whisper API (требует WAV файл)
        - WebSocket отправка (бинарные данные)
        - Стандартный формат для аудио обмена
    
    Алгоритм:
        1. Если audio float32 → конвертируем в int16 через float32_to_int16()
        2. Создаём BytesIO буфер (виртуальный файл в памяти)
        3. Открываем wave.open() в режиме записи ('wb')
        4. Устанавливаем параметры WAV:
           - nchannels = 1 (моно)
           - sampwidth = 2 (int16 = 2 байта)
           - framerate = sample_rate
        5. Записываем audio.tobytes() в WAV
        6. Закрываем wave файл
        7. Возвращаем buffer.getvalue() → bytes
    
    Args:
        audio: Аудио массив (int16 или float32)
        sample_rate: Частота дискретизации (например, 16000)
    
    Returns:
        WAV файл в виде байтов (готов к отправке)
    
    Example:
        >>> audio = np.array([100, -200, 300], dtype=np.int16)
        >>> wav_bytes = audio_to_wav_bytes(audio, 16000)
        >>> print(f"WAV size: {len(wav_bytes)} bytes")
        WAV size: 50 bytes
    """
    # Конвертируем float32 → int16 если нужно
    if audio.dtype == np.float32:
        audio = float32_to_int16(audio)

    # Создаём WAV в памяти
    buffer = BytesIO()
    wav_file: wave.Wave_write = cast(wave.Wave_write, wave.open(buffer, 'wb'))
    with wav_file:
        wav_file.setnchannels(1)            # Моно
        wav_file.setsampwidth(2)            # 2 bytes = int16
        wav_file.setframerate(sample_rate)  # Частота
        wav_file.writeframes(audio.tobytes())

    # Возвращаем bytes
    return buffer.getvalue()



def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Нормализует громкость аудио (peak normalization).
    
    Используется для:
        - Выравнивание громкости между разными источниками
        - Предотвращение клиппинга (искажений)
        - Улучшение качества распознавания Whisper
        - Стандартизация уровня сигнала
    
    Алгоритм:
        1. Находим максимальное абсолютное значение (peak)
        2. Если peak == 0 → возвращаем audio без изменений (тишина)
        3. Если peak > 0 → делим весь массив на peak
        4. Результат: пик громкости становится ±1.0
    
    Args:
        audio: Аудио массив (float32)
    
    Returns:
        Нормализованное аудио (пик = ±1.0)
    
    Formula:
        normalized = audio / max(abs(audio))
    
    Example:
        >>> audio = np.array([0.2, -0.4, 0.6])  # Пик = 0.6
        >>> normalized = normalize_audio(audio)
        >>> print(normalized)
        [0.333 -0.667 1.0]  # Пик теперь 1.0
    """
    # Находим пиковое значение
    peak = np.abs(audio).max()
    
    # Если тишина → не нормализуем
    if peak == 0:
        return audio
    
    # Нормализуем к пику 1.0
    return audio / peak
