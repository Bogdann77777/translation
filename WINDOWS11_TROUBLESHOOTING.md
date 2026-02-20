# Windows 11 Краш - Руководство по решению проблемы

## Проблема
При запуске через Ngrok приложение "чуть-чуть работает и выбивает"

## Возможные причины

### 1. CUDA Out of Memory (OOM)
**Признаки:**
- Приложение крашится через несколько секунд после старта
- В логах нет записей об ошибке (краш на уровне CUDA)

**Решение:**
```yaml
# config.yaml
pipeline:
  batch_queue_size: 3  # Уменьшить с 6 до 3 (меньше батчей в памяти)
```

Или используйте меньшую модель Whisper:
```yaml
models:
  whisper:
    model_size: "medium"  # Вместо large-v3
```

### 2. Ngrok WebSocket Timeout
**Признаки:**
- Соединение обрывается через ~30-60 секунд
- В браузере показывает "disconnected"

**Решение:**
Добавьте в Ngrok keepalive:
```bash
ngrok http 8000 --region us --log stdout
```

### 3. Windows 11 CUDA/PyTorch совместимость
**Признаки:**
- Ошибки типа "CUDA error: device-side assert triggered"
- Работало на Windows 10, не работает на Windows 11

**Решение:**
Обновите PyTorch и CUDA драйвера:
```bash
# Переустановите PyTorch с последней версией CUDA
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Обновите NVIDIA драйвера до последней версии для Windows 11.

### 4. Две GPU и конфликт выделения памяти
**Признаки:**
- У вас две GPU (как указано в config: gpu_id: 0 и gpu_id: 1)
- Краш при переключении между GPU

**Решение:**
Используйте одну GPU для всех моделей:
```yaml
models:
  whisper:
    gpu_id: 0  # Первая GPU

  tts:
    gpu_id: 0  # Та же GPU (вместо 1)
```

## Диагностика

### Шаг 1: Проверьте доступность CUDA
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

### Шаг 2: Проверьте память GPU
```bash
python -c "import torch; print(f'GPU 0 memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB'); print(f'GPU 1 memory: {torch.cuda.get_device_properties(1).total_memory / 1024**3:.1f} GB')"
```

### Шаг 3: Запустите с детальными логами
```bash
# Запустите сервер с уровнем DEBUG
python app/main.py
```

Проверьте файл `logs/session_YYYY-MM-DD.log` на наличие ошибок CUDA.

### Шаг 4: Тест быстрого режима
Попробуйте новый **дословный режим** - он использует меньшие чанки и может быть стабильнее:
1. Запустите приложение
2. Выберите "Дословный переводчик (скорость)" в UI
3. Нажмите Start Session

## Рекомендованная конфигурация для Windows 11

```yaml
# config.yaml - стабильная конфигурация
models:
  whisper:
    model_size: "medium"  # Меньше памяти
    gpu_id: 0

  tts:
    gpu_id: 0  # Та же GPU для избежания конфликтов

pipeline:
  batch_queue_size: 3  # Меньше батчей = меньше VRAM

  vad:
    min_chunk_duration: 8.0   # Уменьшено
    max_phrase_duration: 12.0  # Уменьшено
```

## Если ничего не помогло

1. **Проверьте Ngrok альтернативы:**
   - Localtunnel: `npx localtunnel --port 8000`
   - Cloudflare Tunnel: `cloudflared tunnel --url http://localhost:8000`

2. **Запустите без Ngrok (локально):**
   ```bash
   python app/main.py
   # Откройте http://localhost:8000
   ```

3. **Соберите диагностику и отправьте issue:**
   - Последние 100 строк из logs/
   - Вывод nvidia-smi
   - Версии: Python, PyTorch, CUDA
   - Конфигурация GPU

## Контакты для поддержки
- GitHub Issues: [создайте issue с тегом windows-11]
- Приложите файл логов и вывод диагностических команд
