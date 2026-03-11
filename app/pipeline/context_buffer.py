"""
Модуль управления контекстным буфером для перевода.
Хранит последние N пар (english, russian) для живого нарратива.
"""

import asyncio
from collections import deque
from typing import List, Dict
from app.config import load_config
from app.monitoring.logger import setup_logger


class ContextBuffer:
    """
    Буфер для хранения контекста — пар (EN оригинал, RU перевод).

    Это позволяет модели видеть свой предыдущий русский вывод
    и продолжать нарратив связно, а не переводить каждый чанк как остров.
    """

    def __init__(self):
        """Инициализация буфера."""
        self.config = load_config()["pipeline"]
        self.logger = setup_logger(__name__)

        # Параметры
        self.window_size = self.config["context_window"]
        self.max_chars = self.config["context_max_chars"]

        # Буфер хранит dict {"en": str, "ru": str}
        self.buffer: deque = deque(maxlen=self.window_size)

        # Lock для async операций
        self.lock = asyncio.Lock()

        self.logger.info(f"ContextBuffer initialized: window={self.window_size}, max_chars={self.max_chars}")

    async def add_pair(self, english: str, russian: str) -> None:
        """
        Добавляет пару (английский оригинал, русский перевод) в буфер.

        Args:
            english: Оригинальный английский текст
            russian: Переведённый русский текст
        """
        async with self.lock:
            self.buffer.append({"en": english.strip(), "ru": russian.strip()})
            self.logger.debug(f"Context pair added: EN={english[:40]}... RU={russian[:40]}...")

    async def get_context(self) -> List[Dict[str, str]]:
        """
        Возвращает текущий контекст — список пар {"en": ..., "ru": ...}.

        Обрезает по max_chars (суммарно EN+RU), удаляя самые старые пары.

        Returns:
            List[Dict]: Список пар, от старых к новым
        """
        async with self.lock:
            context = list(self.buffer)

            # Обрезаем по символам если превышает лимит (считаем EN + RU)
            total_chars = sum(len(p["en"]) + len(p["ru"]) for p in context)
            while total_chars > self.max_chars and len(context) > 0:
                removed = context.pop(0)
                total_chars -= len(removed["en"]) + len(removed["ru"])

            return context

    async def clear(self) -> None:
        """Очищает буфер."""
        async with self.lock:
            self.buffer.clear()
            self.logger.info("Context buffer cleared")
