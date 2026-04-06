"""
Smart Buffer: накопитель смысловых единиц между Whisper и LLM.

Принцип работы (модель синхронного переводчика):
- Whisper отдаёт рваные фрагменты (1-4 сек, часто обрезанные на середине слова/фразы)
- SmartBuffer накапливает текст, ищет смысловую границу (. ? ! , ; : —)
- Как только найдена граница → отрезает до неё, отдаёт в LLM
- Остаток остаётся в буфере для следующего цикла
- Если за 2 Whisper-чанка граница не найдена → принудительная отдача

Дополнительно: добавляет метаданные [QUESTION], [REPLY], [DIALOGUE]
для помощи LLM с определением типа разговора.
"""

import re
from app.monitoring.logger import setup_logger


# Символы, на которых можно резать (смысловые границы)
_BOUNDARY_RE = re.compile(r'[.?!;:—]\s*$|,\s+(?=[A-ZА-ЯЁ])')

# Минимальная длина для отдачи (без сильной концовки .?!)
_MIN_EMIT_CHARS = 20

# Слова-продолжения в начале текста → это хвост предыдущего предложения, не отдавать
# NB: and/but/or убраны — это начала клауз, не фрагменты
_CONTINUATION_WORDS = re.compile(
    r'^(to|for|of|from|with|in|on|at|about|like|than|being)\b',
    re.IGNORECASE
)

# Грамматические границы клауз (для текста БЕЗ пунктуации от Whisper)
# Используются как самый низкий приоритет — после всех пунктуационных границ
_CLAUSE_RE = re.compile(
    r'\s+'
    r'(?:'
    # A: Союзы/коннекторы, которые всегда начинают новую клаузу
    r'(?:but|so|yet|because|since|although|however|therefore|meanwhile|otherwise)\s+\w+'
    r'|'
    # B: "and/or" + подлежащее-местоимение (новая клауза, не перечисление)
    r'(?:and|or)\s+(?:I|he|she|they|we|you|it|this|that|there)\s+\w+'
    r'|'
    # C: Местоимение + сокращение (I've, he's, they're, I'd, we'll)
    r"(?:I|he|she|they|we|you|it|that|there)(?:'ve|'d|'ll|'m|'re|'s)\s"
    r'|'
    # D: "I" + частые глаголы устной речи (без сокращения)
    r"I\s+(?:think|know|feel|believe|mean|guess|hope|need|want|just|never|always|"
    r"also|actually|really|don't|didn't|can't|couldn't|wasn't|won't)\b"
    r'|'
    # E: which/who + вспомогательный глагол (относительная клауза)
    r'(?:which|who)\s+(?:is|are|was|were|has|have|had|can|could|will|would|should|might|may)\s+'
    r')',
    re.IGNORECASE
)

# Минимум символов до точки разреза клаузы
_MIN_CLAUSE_EMIT = 15


class SmartBuffer:
    """
    Буфер смысловых единиц между Whisper и LLM.

    Использование:
        buffer = SmartBuffer()
        result = buffer.feed(whisper_text)
        # result = {"text": "...", "metadata": "[QUESTION]"} или None (ещё копим)
    """

    def __init__(self, max_pending_chunks: int = 2):
        """
        Args:
            max_pending_chunks: максимум Whisper-чанков ждать без границы
                                (после этого — принудительная отдача)
        """
        self.logger = setup_logger(__name__)
        self.max_pending_chunks = max_pending_chunks

        # Накопленный текст от Whisper (между отдачами в LLM)
        self._accumulator: str = ""
        # Сколько Whisper-чанков накопилось с последней отдачи
        self._pending_count: int = 0
        # Последний отданный текст (для детекции диалога)
        self._last_emitted: str = ""

    def feed(self, whisper_text: str) -> dict | None:
        """
        Принимает текст от Whisper, решает — отдавать или копить.

        Args:
            whisper_text: текст от Whisper (может быть фрагмент)

        Returns:
            dict {"text": str, "metadata": str} — готовый блок для LLM
            None — ещё копим, ждём следующий чанк
        """
        text = whisper_text.strip()
        if not text:
            return None

        # Добавляем в аккумулятор
        if self._accumulator:
            self._accumulator += " " + text
        else:
            self._accumulator = text
        self._pending_count += 1

        self.logger.debug(
            f"SmartBuffer: +'{text[:60]}' → accumulator={len(self._accumulator)} chars, "
            f"pending={self._pending_count}/{self.max_pending_chunks}"
        )

        # Ищем смысловую границу
        boundary_pos = self._find_boundary(self._accumulator)

        if boundary_pos is not None:
            emit_text = self._accumulator[:boundary_pos].strip()
            remainder = self._accumulator[boundary_pos:].strip()

            # ФИКС B: Текст начинается с continuation word — это обрубок, не отдавать
            # (применяем и min_chars только к фрагментам)
            is_fragment = bool(_CONTINUATION_WORDS.match(emit_text))

            if is_fragment and self._pending_count < self.max_pending_chunks:
                # Фрагмент: ждём и по continuation word, и если слишком короткий
                self.logger.info(
                    f"SmartBuffer: fragment (starts with '{emit_text.split()[0]}', "
                    f"{len(emit_text)} chars) — waiting to merge"
                )
                return None

            # ФИКС A: min_chars ТОЛЬКО для текста без сильной границы (.?!)
            # Законченная фраза с точкой → отдаём сразу, даже если короткая
            has_strong_ending = emit_text.rstrip()[-1:] in '.?!' if emit_text.strip() else False
            if (not has_strong_ending
                    and len(emit_text) < _MIN_EMIT_CHARS
                    and self._pending_count < self.max_pending_chunks):
                self.logger.info(
                    f"SmartBuffer: no strong ending + too short ({len(emit_text)} < {_MIN_EMIT_CHARS}) "
                    f"— waiting for more"
                )
                return None

            self.logger.info(
                f"SmartBuffer: BOUNDARY at pos {boundary_pos} → "
                f"emit {len(emit_text)} chars, keep {len(remainder)} chars"
            )

            self._accumulator = remainder
            self._pending_count = 1 if remainder else 0
            return self._make_result(emit_text)

        # Нет границы — проверяем лимит ожидания
        if self._pending_count >= self.max_pending_chunks:
            # Принудительная отдача (приём "салями")
            emit_text = self._accumulator.strip()
            self.logger.info(
                f"SmartBuffer: FORCED emit after {self._pending_count} chunks → "
                f"{len(emit_text)} chars (no boundary found)"
            )
            self._accumulator = ""
            self._pending_count = 0
            return self._make_result(emit_text)

        # Ещё копим
        self.logger.debug(f"SmartBuffer: waiting for boundary or next chunk...")
        return None

    def flush(self) -> dict | None:
        """
        Принудительно отдаёт всё что есть в буфере.
        Вызывается при остановке сессии.
        """
        if not self._accumulator.strip():
            return None

        emit_text = self._accumulator.strip()
        self._accumulator = ""
        self._pending_count = 0
        self.logger.info(f"SmartBuffer: FLUSH → {len(emit_text)} chars")
        return self._make_result(emit_text)

    def reset(self) -> None:
        """Сбрасывает буфер (новая сессия)."""
        self._accumulator = ""
        self._pending_count = 0
        self._last_emitted = ""

    def _find_boundary(self, text: str) -> int | None:
        """
        Ищет смысловую границу в тексте (приоритет от сильных к слабым):

        1. Конец предложения: . ? !
        2. Разделитель клаузы: ; : —
        3. Запятая перед заглавной буквой
        4. Грамматическая граница клаузы (subject+verb паттерн)

        Returns:
            Позиция для среза, или None
        """
        # Ищем последнюю сильную границу (. ? !)
        last_strong = -1
        for i, ch in enumerate(text):
            if ch in '.?!':
                last_strong = i + 1

        if last_strong > 0:
            # Проверяем что после границы есть хоть что-то существенное
            # или что это конец текста — тогда отдаём всё
            after = text[last_strong:].strip()
            if len(after) < 3:
                # Мало текста после границы — отдаём ВСЁ включая остаток
                return len(text)
            return last_strong

        # Ищем средние границы (; : —)
        last_medium = -1
        for i, ch in enumerate(text):
            if ch in ';:—' and i > 10:  # Минимум 10 символов до границы
                last_medium = i + 1

        if last_medium > 0:
            return last_medium

        # Ищем запятую перед заглавной буквой (новая клауза)
        for match in re.finditer(r',\s+(?=[A-ZА-ЯЁ])', text):
            pos = match.end()
            if pos > 10:  # Минимум 10 символов до запятой
                last_medium = pos

        if last_medium > 0:
            return last_medium

        # 4. Грамматическая граница клаузы (для текста без пунктуации)
        # Ищем ПЕРВЫЙ подходящий паттерн (салями — режем как можно раньше)
        for match in _CLAUSE_RE.finditer(text):
            cut_pos = match.start()  # позиция пробела ПЕРЕД началом клаузы
            if cut_pos >= _MIN_CLAUSE_EMIT:
                self.logger.info(
                    f"SmartBuffer: CLAUSE boundary at pos {cut_pos}: "
                    f"...'{text[max(0,cut_pos-10):cut_pos]}' | "
                    f"'{text[cut_pos:cut_pos+30].strip()}...'"
                )
                return cut_pos

        return None

    def _make_result(self, text: str) -> dict:
        """
        Формирует результат с метаданными.

        Определяет тип фрагмента по пунктуации и контексту:
        - [QUESTION] — вопросительное предложение
        - [REPLY] — короткий ответ после вопроса
        - [DIALOGUE] — прямая речь (кавычки, "said", "asked")
        """
        metadata = ""

        # Определяем вопрос
        if text.rstrip().endswith('?'):
            metadata = "[QUESTION]"

        # Определяем короткий ответ после вопроса
        elif (self._last_emitted.rstrip().endswith('?')
              and len(text.split()) <= 8):
            metadata = "[REPLY]"

        # Определяем прямую речь
        elif re.search(r'[""\'"]', text) or re.search(
                r'\b(said|asked|replied|answered|told|shouted|whispered)\b',
                text, re.IGNORECASE):
            metadata = "[DIALOGUE]"

        self._last_emitted = text

        if metadata:
            self.logger.info(f"SmartBuffer: metadata={metadata}")

        return {"text": text, "metadata": metadata}
