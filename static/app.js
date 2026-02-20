// ============================================
// GLOBALS
// ============================================
let ws = null;
let audioContext = null;
let mediaStream = null;
let scriptNode = null;
let isRecording = false;
let currentSpeed = 1.3;
let translationMode = 'literal'; // 'contextual' or 'literal' (default: literal for fast response)

// ============================================
// LOGGING TO HTML CONSOLE
// ============================================
function addLog(level, message) {
    const logConsole = document.getElementById('logConsole');
    if (!logConsole) return;

    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false, fractionalSecondDigits: 3 });
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.innerHTML = `
        <span class="log-timestamp">[${timestamp}]</span>
        <span class="log-level ${level}">${level.toUpperCase()}</span>
        <span class="log-message">${escapeHtml(String(message))}</span>
    `;

    logConsole.appendChild(logEntry);
    logConsole.scrollTop = logConsole.scrollHeight; // Auto-scroll to bottom
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Override console methods to also log to HTML
const originalLog = console.log;
const originalWarn = console.warn;
const originalError = console.error;

console.log = function(...args) {
    originalLog.apply(console, args);
    addLog('info', args.join(' '));
};

console.warn = function(...args) {
    originalWarn.apply(console, args);
    addLog('warn', args.join(' '));
};

console.error = function(...args) {
    originalError.apply(console, args);
    addLog('error', args.join(' '));
};

// ============================================
// WEBSOCKET CONNECTION
// ============================================
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/translate`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('[WS] WebSocket connected to', wsUrl);
        updateStatus('connected');
    };

    ws.onmessage = (event) => {
        console.log('[WS] Message received:', event.data.substring(0, 200));
        const message = JSON.parse(event.data);
        console.log('[WS] Parsed message type:', message.type);
        handleMessage(message);
    };

    ws.onerror = (error) => {
        console.error('[WS] WebSocket error:', error);
        updateStatus('error');
    };

    ws.onclose = (event) => {
        console.log('[WS] WebSocket disconnected. Code:', event.code, 'Reason:', event.reason);
        updateStatus('disconnected');
        
        // Stop recording if active
        if (isRecording) {
            console.log('[WS] Connection lost while recording - stopping session');
            stopSession();
        }

        // Переподключение через 3 секунды
        console.log('[WS] Reconnecting in 3 seconds...');
        setTimeout(connectWebSocket, 3000);
    };
}

function updateStatus(status) {
    const statusEl = document.getElementById('status');
    statusEl.className = `status-${status}`;
    statusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);
}

// ============================================
// MESSAGE HANDLING
// ============================================
function handleMessage(message) {
    const { type, data } = message;

    switch(type) {
        case 'session_started':
            console.log('[MSG] Session started');
            break;

        case 'transcription':
            console.log('[MSG] Transcription received:', message.text.substring(0, 100));
            appendTranscript('englishText', message.text);
            break;

        case 'translation':
            console.log('[MSG] Translation received:', message.translated.substring(0, 100));
            appendTranscript('russianText', message.translated);
            break;

        case 'audio_output':
            console.log('[MSG] Audio output received:', message.data.length, 'bytes (base64)');
            playAudio(message.data);
            break;

        case 'metrics':
            console.log('[MSG] Metrics update:', JSON.stringify(data).substring(0, 150));
            updateMetrics(data);
            break;

        case 'error':
            console.error('[MSG] Server error:', message.message);
            break;

        default:
            console.warn('[MSG] Unknown message type:', type);
    }
}

function appendTranscript(elementId, text) {
    const el = document.getElementById(elementId);
    el.textContent += text + '\n\n';
    el.scrollTop = el.scrollHeight;
}

function playAudio(base64Audio) {
    // Декодируем base64 в ArrayBuffer
    const binaryString = atob(base64Audio);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    
    // Создаём AudioContext если нужно
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    // Декодируем WAV
    audioContext.decodeAudioData(bytes.buffer, (audioBuffer) => {
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        source.start(0);
    }).catch(err => {
        console.error('Audio decode error:', err);
    });
}

function updateMetrics(metrics) {
    // Latency метрики
    if (metrics.latency) {
        document.getElementById('latencyStt').textContent = 
            (metrics.latency.stt || 0).toFixed(2) + 's';
        document.getElementById('latencyTranslation').textContent = 
            (metrics.latency.translation || 0).toFixed(2) + 's';
        document.getElementById('latencyTts').textContent = 
            (metrics.latency.tts || 0).toFixed(2) + 's';
        document.getElementById('latencyE2e').textContent = 
            (metrics.latency.e2e || 0).toFixed(2) + 's';
    }
    
    // Batches processed
    document.getElementById('batchCount').textContent = 
        metrics.batches_processed || 0;
    
    // Uptime
    document.getElementById('uptime').textContent = 
        (metrics.uptime || 0).toFixed(0) + 's';
    
    // Обновляем слоты
    ['slot1', 'slot2', 'slot3'].forEach((id, i) => {
        const slotEl = document.getElementById(id);
        slotEl.className = 'slot';
        const slotData = metrics.slots ? metrics.slots.find(s => s.slot === i + 1) : null;
        if (slotData) {
            slotEl.classList.add(slotData.status);
        }
    });
}

// ============================================
// AUDIO CAPTURE
// ============================================
async function startSession() {
    try {
        console.log('[AUDIO] Requesting microphone access...');

        // Check if mediaDevices is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            // Fallback for older browsers or insecure context
            const getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

            if (!getUserMedia) {
                throw new Error('Your browser does not support microphone access. Use Chrome/Firefox on HTTPS or localhost.');
            }

            // Use legacy API with promise wrapper
            mediaStream = await new Promise((resolve, reject) => {
                getUserMedia.call(navigator,
                    { audio: true },
                    resolve,
                    reject
                );
            });
        } else {
            // Modern API - use selected microphone if available
            const audioConstraints = {
                echoCancellation: true,
                noiseSuppression: true
            };

            // Add deviceId constraint if a specific microphone is selected
            if (selectedMicId) {
                audioConstraints.deviceId = { exact: selectedMicId };
                console.log('[AUDIO] Using selected microphone:', selectedMicId);
            }

            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: audioConstraints
            });
        }

        console.log('[AUDIO] Microphone access granted');

        // Создаём AudioContext
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });

        console.log('[AUDIO] AudioContext created, sample rate:', audioContext.sampleRate);

        // Создаём источник из потока
        const source = audioContext.createMediaStreamSource(mediaStream);

        // Создаём ScriptProcessorNode для обработки аудио
        scriptNode = audioContext.createScriptProcessor(4096, 1, 1);

        let chunkCount = 0;

        scriptNode.onaudioprocess = (event) => {
            if (!isRecording) return;

            chunkCount++;

            // Получаем аудио данные (Float32Array)
            const inputData = event.inputBuffer.getChannelData(0);

            // Конвертируем в Int16Array (для эффективной передачи)
            const int16Data = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                const s = Math.max(-1, Math.min(1, inputData[i]));
                int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }

            // Отправляем на сервер (base64)
            const base64Audio = btoa(String.fromCharCode.apply(null, new Uint8Array(int16Data.buffer)));

            if (chunkCount % 10 === 1) { // Log every 10th chunk
                console.log('[AUDIO] Sending chunk #' + chunkCount + ', size:', int16Data.length, 'samples');
            }

            ws.send(JSON.stringify({
                type: 'audio',
                data: base64Audio
            }));
        };

        // Подключаем nodes
        source.connect(scriptNode);
        scriptNode.connect(audioContext.destination);

        console.log('[AUDIO] Audio pipeline connected');

        // Отправляем команду старта на сервер с режимом перевода и темой
        const topic = document.getElementById('topicInput').value.trim();
        console.log('[AUDIO] Sending "start" command to server (mode:', translationMode, 'topic:', topic || 'none' + ')');
        ws.send(JSON.stringify({
            type: 'start',
            mode: translationMode,
            topic: topic || null
        }));

        // Обновляем UI
        isRecording = true;
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;

        console.log('[AUDIO] Recording started successfully');

    } catch (error) {
        console.error('[AUDIO] Failed to start session:', error.name, error.message);

        // Show detailed error
        let errorMsg = 'Microphone error: ' + error.name;
        if (error.name === 'NotAllowedError') {
            errorMsg = 'Microphone access denied. Click the lock icon in address bar to allow.';
        } else if (error.name === 'NotFoundError') {
            errorMsg = 'No microphone found. Please connect a microphone.';
        } else if (error.name === 'NotReadableError') {
            errorMsg = 'Microphone is busy (used by another app). Close other apps using mic.';
        } else if (error.name === 'OverconstrainedError') {
            errorMsg = 'Microphone does not support requested settings.';
        }

        alert(errorMsg);
    }
}

// ============================================
// STOP SESSION
// ============================================
function stopSession() {
    console.log('[AUDIO] Stopping session...');

    // Останавливаем запись
    isRecording = false;

    // Останавливаем media stream
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
        console.log('[AUDIO] Media stream stopped');
    }

    // Отключаем audio nodes
    if (scriptNode) {
        scriptNode.disconnect();
        scriptNode = null;
        console.log('[AUDIO] Script node disconnected');
    }

    if (audioContext) {
        audioContext.close();
        audioContext = null;
        console.log('[AUDIO] Audio context closed');
    }

    // Отправляем команду остановки на сервер
    if (ws && ws.readyState === WebSocket.OPEN) {
        console.log('[AUDIO] Sending "stop" command to server');
        ws.send(JSON.stringify({ type: 'stop' }));
    } else {
        console.warn('[AUDIO] Cannot send stop command - WebSocket not open');
    }

    // Обновляем UI
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;

    console.log('[AUDIO] Session stopped');
    
    console.log('Recording stopped');
}

// ============================================
// MICROPHONE SELECTION
// ============================================
let selectedMicId = null;

async function loadMicrophones() {
    const micSelect = document.getElementById('micSelect');

    try {
        console.log('[MIC] Requesting permission to enumerate devices...');

        // First, request permission (needed to get device labels)
        await navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                // Stop the stream immediately - we just needed permission
                stream.getTracks().forEach(track => track.stop());
            });

        // Now enumerate devices
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(device => device.kind === 'audioinput');

        console.log('[MIC] Found', audioInputs.length, 'audio input devices');

        micSelect.innerHTML = '';

        if (audioInputs.length === 0) {
            micSelect.innerHTML = '<option value="">No microphones found</option>';
            return;
        }

        audioInputs.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Microphone ${index + 1}`;
            micSelect.appendChild(option);
        });

        // Select first device by default
        selectedMicId = audioInputs[0].deviceId;
        console.log('[MIC] Default selected:', audioInputs[0].label || 'Microphone 1');

    } catch (error) {
        console.error('[MIC] Failed to enumerate devices:', error);
        micSelect.innerHTML = '<option value="">Error loading microphones</option>';
    }
}

// ============================================
// VOICE SELECTION
// ============================================
async function loadVoices() {
    try {
        console.log('[VOICE] Loading available voices...');
        const response = await fetch('/voices');
        const data = await response.json();

        const voiceSelect = document.getElementById('voiceSelect');
        voiceSelect.innerHTML = ''; // Clear loading text

        if (data.voices && data.voices.length > 0) {
            data.voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.path;
                option.textContent = voice.name;

                // Mark current voice as selected
                if (data.current && voice.path === data.current) {
                    option.selected = true;
                }

                voiceSelect.appendChild(option);
            });
            console.log('[VOICE] Loaded', data.voices.length, 'voices');
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No voices found';
            voiceSelect.appendChild(option);
            console.warn('[VOICE] No voice samples found in voice_samples/');
        }
    } catch (error) {
        console.error('[VOICE] Failed to load voices:', error);
        const voiceSelect = document.getElementById('voiceSelect');
        voiceSelect.innerHTML = '<option value="">Error loading</option>';
    }
}

// ============================================
// EVENT LISTENERS
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    // Подключаем WebSocket
    connectWebSocket();

    // Load available voices
    loadVoices();

    // Привязываем кнопки
    document.getElementById('startBtn').addEventListener('click', startSession);
    document.getElementById('stopBtn').addEventListener('click', stopSession);
    document.getElementById('clearLogsBtn').addEventListener('click', () => {
        const logConsole = document.getElementById('logConsole');
        logConsole.innerHTML = '';
        console.log('Logs cleared');
    });

    // Speed slider
    const speedSlider = document.getElementById('speedSlider');
    const speedValue = document.getElementById('speedValue');

    speedSlider.addEventListener('input', () => {
        currentSpeed = parseFloat(speedSlider.value);
        speedValue.textContent = currentSpeed.toFixed(1) + 'x';

        // Send speed update to server
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'set_speed',
                speed: currentSpeed
            }));
            console.log('[SPEED] Updated to', currentSpeed + 'x');
        }
    });

    // Voice selector
    const voiceSelect = document.getElementById('voiceSelect');
    voiceSelect.addEventListener('change', () => {
        const selectedVoice = voiceSelect.value;
        if (selectedVoice && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'set_voice',
                voice: selectedVoice
            }));
            console.log('[VOICE] Switched to:', selectedVoice);
        }
    });

    // Microphone selector
    const micSelect = document.getElementById('micSelect');
    micSelect.addEventListener('change', () => {
        selectedMicId = micSelect.value;
        console.log('[MIC] Selected microphone:', micSelect.options[micSelect.selectedIndex].text);
    });

    // Refresh microphones button
    document.getElementById('refreshMicBtn').addEventListener('click', () => {
        console.log('[MIC] Refreshing microphone list...');
        loadMicrophones();
    });

    // Load microphones on page load
    loadMicrophones();

    // Mode selector
    const modeSelect = document.getElementById('modeSelect');
    modeSelect.addEventListener('change', () => {
        translationMode = modeSelect.value;
        console.log('[MODE] Switched to:', translationMode);

        // Disable mode change during active session
        if (isRecording) {
            console.warn('[MODE] Cannot change mode during active session');
            alert('Please stop the current session before changing translation mode');
            // Revert to previous value
            modeSelect.value = translationMode === 'contextual' ? 'literal' : 'contextual';
            translationMode = modeSelect.value;
        }
    });

    console.log('App initialized');
});
