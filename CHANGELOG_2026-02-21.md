# CHANGELOG - 2026-02-21 Session

## Изменения для быстрого отката

### 1. WebSocket Timeout (УБРАНЫ таймауты)
**Файл:** `app/main.py:282-287`

**Было:**
```python
uvicorn.run(
    app,
    host=server_config["host"],
    port=server_config["port"],
    log_level="info"
)
```

**Стало:**
```python
uvicorn.run(
    app,
    host=server_config["host"],
    port=server_config["port"],
    log_level="info",
    timeout_keep_alive=86400,  # 24 hours - effectively unlimited
    ws_ping_interval=None,     # Disable auto-ping
    ws_ping_timeout=None       # No timeout
)
```

**Причина:** Фильмы могут иметь паузы 5-10 минут, не должно быть автоматического disconnect.

---

### 2. Блокировка не-английских языков
**Файл:** `app/pipeline/batch_queue.py:406-425`

**Было:**
```python
if detected_lang in ["ru", "russian", "rus"]:
    # Block only Russian
```

**Стало:**
```python
if detected_lang not in ["en", "english"]:
    # Block ALL non-English languages
```

**Причина:** Разговоры на русском/других языках не должны попадать в переводчик.

---

### 3. Adaptive Speed (AUTO-ADJUSTMENT)
**Файл:** `config.yaml:31-37`

**Добавлено:**
```yaml
speed: 1.3  # Base speed
adaptive_speed:
  enabled: true
  min_speed: 1.3
  max_speed: 2.0
  queue_threshold_low: 2
  queue_threshold_high: 5
```

**Файл:** `app/pipeline/batch_queue.py:80-98` (initialization)
```python
# ADAPTIVE SPEED config loading
tts_config = load_config()["models"]["tts"]
self.adaptive_speed_config = tts_config.get("adaptive_speed", {})
self.adaptive_speed_enabled = self.adaptive_speed_config.get("enabled", False)
```

**Файл:** `app/pipeline/batch_queue.py:155-189` (calculation method)
```python
def _calculate_adaptive_speed(self, queue_size: int) -> float:
    """Auto-adjust TTS speed based on queue size."""
    # Linear interpolation between min_speed and max_speed
    # If queue grows → speed up to prevent lag accumulation
```

**Файл:** `app/pipeline/batch_queue.py:491-505` (integration)
```python
# Before TTS synthesis:
if self.adaptive_speed_enabled:
    queue_size = self.ready_queue.qsize()
    adaptive_speed = self._calculate_adaptive_speed(queue_size)
    self.xtts_engine.speed = adaptive_speed
```

**Причина:** Если спикер говорит быстро, TTS не успевает → очередь растёт → автоматически ускоряем.

---

### 4. Queue Overflow Protection
**Файл:** `config.yaml:42`

**Добавлено:**
```yaml
max_ready_queue_size: 10  # Maximum chunks in ready queue
```

**Файл:** `app/pipeline/batch_queue.py:81`

**Было:**
```python
self.ready_queue = asyncio.Queue()  # Unlimited queue
```

**Стало:**
```python
self.max_ready_queue_size = self.config.get("max_ready_queue_size", 10)
self.ready_queue = asyncio.Queue()  # Queue with overflow protection
```

**Файл:** `app/pipeline/batch_queue.py:218-241` (overflow logic)

**Добавлено:**
```python
# QUEUE OVERFLOW PROTECTION
current_queue_size = self.ready_queue.qsize()
if current_queue_size >= self.max_ready_queue_size:
    # Skip old chunk
    old_batch = self.ready_queue.get_nowait()
    self.logger.warning(f"⚠️ Queue OVERFLOW - SKIPPED old chunk")
```

**Причина:** Предотвращает 30-секундное отставание после паузы YouTube. Если очередь >= 10 чанков, старые отбрасываются.

---

### 5. Emergency Fat Chunk Protection
**Файл:** `app/pipeline/literal_stream_processor.py:109-120`

**Добавлено после строки 107:**
```python
# EMERGENCY STOP: If chunk > 10s, finalize immediately
if phrase_duration >= 10.0:
    self.logger.warning(
        f"⚠️ EMERGENCY: Fat chunk detected ({phrase_duration:.1f}s) - "
        f"forcing finalization NOW!"
    )
    should_finalize = True
```

**Причина:** Защита от "жирных чанков" (> 10s) из-за багов VAD или непрерывной речи.

---

### 6. TTS Semaphore Fix (CUDA Race Condition)
**Файл:** `app/pipeline/batch_queue.py:95-101`

**Было (ошибка):**
```python
self.tts_semaphore = asyncio.Semaphore(2)  # Parallel TTS → CUDA errors
```

**Стало:**
```python
self.tts_semaphore = asyncio.Semaphore(1)  # Sequential TTS (CUDA race fix)
```

**Причина:** Два параллельных вызова `self.model.tts()` на ОДНОЙ модели → "device-side assert triggered". Sequential (semaphore=1) решает проблему.

---

## Как откатить изменения

### Полный откат через git:
```bash
git checkout HEAD -- app/main.py app/pipeline/batch_queue.py app/pipeline/literal_stream_processor.py config.yaml
```

### Частичный откат (по функции):

**Отключить Adaptive Speed:**
```yaml
# config.yaml:37
adaptive_speed:
  enabled: false  # ← Изменить на false
```

**Отключить Queue Overflow Protection:**
```yaml
# config.yaml:42
max_ready_queue_size: 999  # ← Большое число = практически unlimited
```

**Вернуть WebSocket timeout:**
```python
# app/main.py:282-287 - убрать параметры:
uvicorn.run(
    app,
    host=server_config["host"],
    port=server_config["port"],
    log_level="info"
    # Удалить: timeout_keep_alive, ws_ping_interval, ws_ping_timeout
)
```

**Вернуть блокировку только русского:**
```python
# app/pipeline/batch_queue.py:408
if detected_lang in ["ru", "russian", "rus"]:  # Вместо: not in ["en", "english"]
```

---

## Тестирование

### Что проверить:
1. ✅ **CUDA errors ушли** (tts_semaphore=1)
2. ✅ **Нет disconnect** при длинных паузах (5+ минут)
3. ✅ **Adaptive speed работает** - смотри логи "🎚️ Adaptive speed: 1.3x → 1.8x"
4. ✅ **Queue overflow protection** - смотри логи "⚠️ Queue OVERFLOW"
5. ✅ **Fat chunks обнаружены** - смотри логи "⚠️ EMERGENCY: Fat chunk"
6. ✅ **Русская речь блокируется** - смотри логи "⛔ Non-English speech detected"

### Метрики для мониторинга:
- **Queue size:** `/status` endpoint - показывает текущий размер очереди
- **TTS speed:** Логи - "Adaptive speed: X.Xx"
- **Latency:** `/metrics` endpoint - e2e, tts, stt latency

---

## Git Commit для отката

```bash
# Если нужно откатиться к версии ПЕРЕД этими изменениями:
git log --oneline  # Найти commit ID перед изменениями
git checkout <commit-id> -- .
```

**Current HEAD перед изменениями:** (смотри `git log` для точного commit ID)

---

## Заметки

- Все изменения **обратно совместимы** - если отключить adaptive_speed.enabled=false, работает как раньше
- TTS semaphore=1 КРИТИЧНО для стабильности - не менять обратно на 2!
- WebSocket timeout можно вернуть если нужно, но для фильмов лучше unlimited
- Queue overflow protection можно настроить через max_ready_queue_size (10 по умолчанию)

---

**Дата:** 2026-02-21
**Автор:** Claude Sonnet 4.5
**Версия:** Session with adaptive speed + queue overflow protection + fat chunk protection
