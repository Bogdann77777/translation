"""
Модуль для работы с OpenRouter API (Translation LLM).
"""

import asyncio
from openai import AsyncOpenAI
from typing import List, Dict, Any
from app.config import load_config, get_api_key
from app.monitoring.logger import setup_logger


class OpenRouterClient:
    """
    Клиент для OpenRouter API (Claude 3.5 Sonnet).
    
    Функционал:
        - Перевод текста с English → Russian
        - Учёт контекста (предыдущие фразы)
        - Streaming (для будущих улучшений)
    """
    
    def __init__(self):
        """Инициализация клиента."""
        self.config = load_config()["models"]["translation"]
        self.logger = setup_logger(__name__)
        
        # API клиент
        api_key = get_api_key(self.config["api_key_env"])
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        
        # Параметры
        self.model = self.config["model"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        
        self.logger.info(f"OpenRouter client initialized: {self.model}")
    
    async def translate(self, text: str, context: List[str] = None, topic: str = None) -> str:
        """
        Переводит текст с English → Russian с retry логикой.

        Args:
            text: Английский текст для перевода
            context: Список предыдущих фраз (для контекста)
            topic: Опциональная тема/контекст разговора (для улучшения точности)

        Returns:
            str: Переведённый текст на русском

        Алгоритм:
            1. Формируем system prompt
            2. Добавляем тему если есть
            3. Добавляем контекст если есть
            4. Вызываем LLM с exponential backoff
            5. При ошибке повторяем до max_attempts
            6. Возвращаем перевод
        """
        # SEMANTIC TRANSLATION prompt - понимание СМЫСЛА, не слов!
        system_prompt = (
            "You are a PROFESSIONAL INTERPRETER translating live speech from English to Russian.\n\n"
            "CRITICAL RULES:\n"
            "1. UNDERSTAND THE MEANING and CONTEXT before translating - don't translate word-by-word!\n"
            "2. Previous sentences show the CONVERSATION FLOW - use them to understand what the speaker means.\n"
            "3. Translate the INTENT and MEANING, not just literal words.\n"
            "4. Use natural, spoken Russian - как говорят люди, не как пишут в книгах.\n"
            "5. Keep it CONCISE - same length or shorter than original.\n"
            "6. If someone speaks informally/casually, translate informally too.\n"
            "7. PROFANITY: Replace with euphemisms ('черт', 'твою мать', 'блин', etc) - keep emotion, not words.\n\n"
            "Output ONLY the Russian translation, no explanations, no formatting."
        )
        
        # Формируем user message
        user_message = f"Translate to Russian: {text}"

        # Добавляем тему если указана (помогает понять общий контекст разговора)
        if topic:
            user_message = f"Topic/Theme: {topic}\n\n{user_message}"

        # Добавляем контекст если есть - показываем ПОТОК РАЗГОВОРА для понимания смысла
        if context and len(context) > 0:
            # Нумеруем предложения чтобы показать ПОСЛЕДОВАТЕЛЬНОСТЬ мысли
            context_lines = [f"{i+1}. {sentence}" for i, sentence in enumerate(context)]
            context_str = "\n".join(context_lines)
            user_message = (
                f"CONVERSATION HISTORY (understand the topic and flow of thought):\n"
                f"{context_str}\n\n"
                f"NOW TRANSLATE THIS (based on context above):\n{user_message}"
            )
        
        # Retry параметры из конфига
        retry_config = load_config()["pipeline"]["retry"]
        max_attempts = retry_config["max_attempts"]
        backoff_factor = retry_config["backoff_factor"]
        
        # Пытаемся с retry
        for attempt in range(max_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                translation = response.choices[0].message.content.strip()
                finish_reason = getattr(response.choices[0], 'finish_reason', None)

                # Проверяем если перевод обрезан
                if finish_reason == "length":
                    self.logger.warning(f"TRANSLATION TRUNCATED! max_tokens={self.max_tokens} not enough. Input: {len(text)} chars")

                self.logger.info(f"Translated: {len(text)} -> {len(translation)} chars")
                return translation
                
            except Exception as e:
                wait_time = backoff_factor ** attempt
                self.logger.warning(
                    f"OpenRouter API error (attempt {attempt + 1}/{max_attempts}): {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                
                if attempt < max_attempts - 1:
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"OpenRouter API failed after {max_attempts} attempts")
                    raise
