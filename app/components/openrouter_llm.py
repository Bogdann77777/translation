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
    Клиент для OpenRouter API.

    Использует multi-turn формат: модель видит предыдущие EN→RU пары
    как живой диалог и продолжает нарратив связно.
    """

    def __init__(self):
        """Инициализация клиента."""
        self.config = load_config()["models"]["translation"]
        self.logger = setup_logger(__name__)

        api_key = get_api_key(self.config["api_key_env"])
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        self.model = self.config["model"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]

        self.logger.info(f"OpenRouter client initialized: {self.model}")

    async def translate(self, text: str, context: List[Dict[str, str]] = None, topic: str = None) -> str:
        """
        Переводит текст EN → RU используя multi-turn формат.

        Модель видит предыдущие переводы как настоящий диалог (user/assistant),
        что позволяет ей:
        - Продолжать нарратив связно (не остров)
        - Поддерживать стиль и регистр из предыдущих чанков
        - Понимать незавершённые мысли из контекста

        Args:
            text: Английский текст для перевода
            context: Список пар {"en": ..., "ru": ...} — предыдущие переводы
            topic: Опциональная тема разговора

        Returns:
            str: Переведённый текст на русском
        """
        system_prompt = (
            "You are a PROFESSIONAL SIMULTANEOUS INTERPRETER translating live speech to Russian.\n\n"
            "CRITICAL RULES:\n"
            "1. Translate ALL languages to Russian — English, French, Spanish, German, any language.\n"
            "2. FAITHFUL translation: stay close to the original words and structure.\n"
            "   DO NOT paraphrase, DO NOT invent content, DO NOT summarize.\n"
            "3. Only adapt where a literal translation would be incomprehensible in Russian:\n"
            "   - Idioms → translate to the Russian equivalent idiom (not word-by-word)\n"
            "   - Ambiguous phrases → use context to pick the right meaning\n"
            "4. You are CONTINUING an ongoing translation — use previous exchanges to understand context.\n"
            "5. If text is a fragment mid-thought: translate ONLY what is written. DO NOT complete the sentence.\n"
            "   Short fragment → short translation. Never expand beyond what was said.\n"
            "6. Keep the SAME register/style as previous translations. Same length as original.\n"
            "   Output flows as continuous narration — not disconnected fragments.\n"
            "7. Natural spoken Russian grammar, but faithful to what was said.\n"
            "8. PROFANITY: Replace with Russian euphemisms ('чёрт', 'блин', 'твою мать') — keep emotion.\n"
            "9. INPUT IS SPEECH-TO-TEXT: May contain recognition errors (a wrong word that sounds similar\n"
            "   to the real one). If 1-2 words clearly do not fit the grammar or meaning of the sentence —\n"
            "   silently correct to the most likely intended word using surrounding context and previous\n"
            "   translations. Do NOT flag corrections. Do NOT guess wildly — if unsure, translate as-is.\n"
            "   IMPORTANT: A capitalized word at the START of a chunk (e.g. 'Baker', 'Driver', 'Officer')\n"
            "   is almost certainly a common noun, NOT a person's name — the article was dropped by STT.\n"
            "   Translate as the profession/role: Baker → пекарь, Driver → водитель, Officer → офицер.\n\n"
            "Output ONLY the Russian translation, no explanations, no formatting."
        )

        # Формируем один user message (не multi-turn — меньше overhead в API)
        user_message = ""

        if topic:
            user_message += f"Session topic: {topic}\n\n"

        if context:
            context_lines = []
            for i, pair in enumerate(context):
                context_lines.append(f"{i+1}. EN: {pair['en']}\n   RU: {pair['ru']}")
            user_message += "RECENT CONTEXT (EN originals + your previous RU translations):\n"
            user_message += "\n".join(context_lines)
            user_message += "\n\n"

        user_message += f"NOW TRANSLATE TO RUSSIAN:\n{text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Retry параметры
        retry_config = load_config()["pipeline"]["retry"]
        max_attempts = retry_config["max_attempts"]
        backoff_factor = retry_config["backoff_factor"]

        for attempt in range(max_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                translation = response.choices[0].message.content.strip()
                finish_reason = getattr(response.choices[0], 'finish_reason', None)

                if finish_reason == "length":
                    self.logger.warning(
                        f"TRANSLATION TRUNCATED! max_tokens={self.max_tokens} not enough. "
                        f"Input: {len(text)} chars, context pairs: {len(context) if context else 0}"
                    )

                context_count = len(context) if context else 0
                self.logger.info(
                    f"Translated ({context_count} context pairs): "
                    f"{len(text)} -> {len(translation)} chars"
                )
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
