"""
Модуль управления контекстным буфером для перевода.
Хранит последние N предложений для улучшения качества перевода.
"""

import asyncio
from collections import deque
from typing import List
from app.config import load_config
from app.monitoring.logger import setup_logger


class ContextBuffer:
    """
    Буфер для хранения контекста (предыдущих предложений).
    
    Функционал:
        - Хранит последние N предложений
        - Обрезает по количеству и по символам
        - Thread-safe операции
    """
    
    def __init__(self):
        """Инициализация буфера."""
        self.config = load_config()["pipeline"]
        self.logger = setup_logger(__name__)
        
        # Параметры
        self.window_size = self.config["context_window"]
        self.max_chars = self.config["context_max_chars"]
        
        # Буфер (FIFO очередь)
        self.buffer = deque(maxlen=self.window_size)
        
        # Lock для async операций
        self.lock = asyncio.Lock()
        
        self.logger.info(f"ContextBuffer initialized: window={self.window_size}")
    
    async def add_sentence(self, sentence: str) -> None:
        """
        Добавляет предложение в буфер.
        
        Args:
            sentence: Текст предложения
        
        Алгоритм:
            1. Блокируем через async lock
            2. Добавляем в deque (автоматически удалит старое если full)
            3. Разблокируем
        """
        async with self.lock:
            self.buffer.append(sentence)
            self.logger.debug(f"Added to context: {sentence[:50]}...")
    
    async def get_context(self) -> List[str]:
        """
        Возвращает текущий контекст (список предложений).
        
        Returns:
            List[str]: Последние предложения
        
        Алгоритм:
            1. Блокируем через async lock
            2. Копируем buffer в list
            3. Обрезаем по max_chars если нужно
            4. Возвращаем результат
        """
        async with self.lock:
            context = list(self.buffer)
            
            # Обрезаем по символам если превышает лимит
            total_chars = sum(len(s) for s in context)
            if total_chars > self.max_chars:
                # Удаляем старые предложения пока не влезем
                while total_chars > self.max_chars and len(context) > 0:
                    removed = context.pop(0)
                    total_chars -= len(removed)
            
            return context
    
    async def clear(self) -> None:
        """Очищает буфер."""
        async with self.lock:
            self.buffer.clear()
            self.logger.info("Context buffer cleared")
