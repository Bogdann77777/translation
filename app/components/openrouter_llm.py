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
    
    async def translate(self, text: str, context: List[str] = None) -> str:
        """
        Переводит текст с English → Russian с retry логикой.
        
        Args:
            text: Английский текст для перевода
            context: Список предыдущих фраз (для контекста)
        
        Returns:
            str: Переведённый текст на русском
        
        Алгоритм:
            1. Формируем system prompt
            2. Добавляем контекст если есть
            3. Вызываем LLM с exponential backoff
            4. При ошибке повторяем до max_attempts
            5. Возвращаем перевод
        """
        # System prompt for contextual, natural translation
        system_prompt = (
            "You are a professional translator specializing in natural, contextual translation. "
            "Your task is to translate English to Russian by conveying the MEANING and INTENT, not word-for-word. "
            "\n\nIMPORTANT RULES:\n"
            "1. Remove filler words (like, you know, I mean, um, basically, etc.) - don't translate them\n"
            "2. Translate the MEANING in natural Russian, not literal words\n"
            "3. Adapt idioms and slang to Russian equivalents\n"
            "4. Make the Russian text sound natural, as if originally spoken in Russian\n"
            "5. Keep the same tone and emotion, but use Russian speech patterns\n"
            "6. Simplify overly complex or redundant phrases\n"
            "7. CRITICAL: If context is provided, use it ONLY for understanding. Do NOT translate or repeat context sentences in your output\n"
            "8. Translate ONLY the sentence after 'Translate to Russian:', nothing else\n"
            "\nProvide ONLY the clean Russian translation of the NEW sentence, no explanations, no repeated context."
        )
        
        # Формируем user message
        user_message = f"Translate to Russian: {text}"
        
        # Добавляем контекст если есть
        if context and len(context) > 0:
            context_str = "\n".join(f"- {sentence}" for sentence in context[-5:])
            user_message = f"Context (previous sentences):\n{context_str}\n\n{user_message}"
        
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
