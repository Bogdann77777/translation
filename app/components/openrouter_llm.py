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
        # System prompt for LITERAL, word-by-word translation with smart context awareness
        system_prompt = (
            "You are a professional translator specializing in LITERAL, accurate translation. "
            "Your task is to translate English to Russian as WORD-FOR-WORD as possible, while preserving meaning. "
            "\n\nCRITICAL RULES (STRICTLY FOLLOW):\n"
            "1. DEFAULT: Translate LITERALLY, word-by-word, preserving sentence structure\n"
            "2. ONLY change words/phrases if they are grammatically impossible or completely nonsensical in Russian\n"
            "3. Keep filler words (like, you know, um, basically) - translate them naturally (типа, знаешь, эм, в общем)\n"
            "4. Idioms and phraseological units: translate to Russian equivalents ONLY if direct translation is unclear\n"
            "5. Preserve the speaker's exact wording and style - don't rephrase or simplify\n"
            "6. Use context ONLY to disambiguate unclear words (1-2 words max), NOT to rephrase entire sentences\n"
            "7. Do NOT remove, add, or significantly change the meaning - stay as close to original as possible\n"
            "8. If the speaker made a grammatical error or spoke awkwardly, preserve that awkwardness in translation\n"
            "9. CRITICAL: If context is provided, use it ONLY for understanding. Do NOT translate or repeat context in output\n"
            "10. Translate ONLY the sentence after 'Translate to Russian:', nothing else\n"
            "\nThink like a movie subtitle translator: accurate, literal, but still comprehensible. "
            "Provide ONLY the clean Russian translation, no explanations, no repeated context."
        )
        
        # Формируем user message
        user_message = f"Translate to Russian: {text}"

        # Добавляем тему если указана (помогает понять общий контекст разговора)
        if topic:
            user_message = f"Topic/Theme: {topic}\n\n{user_message}"

        # Добавляем контекст если есть (используем ВСЕ предложения из буфера, не только последние 5)
        if context and len(context) > 0:
            context_str = "\n".join(f"- {sentence}" for sentence in context)
            user_message = f"Context (previous sentences for understanding topic/flow):\n{context_str}\n\n{user_message}"
        
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
