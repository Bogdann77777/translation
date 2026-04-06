"""
Qwen3 TTS persistent daemon — runs inside qwen3tts_env.

Loads model ONCE, then listens on a TCP socket for synthesis requests.
Survives server restarts — on reconnect the model is already in VRAM.

Protocol (TCP, one persistent connection per client):
  Client → Server: {"text": "...", "ref_audio": "...", "ref_text": ""}\n
  Server → Client: {"wav_b64": "...", "sr": 24000}\n  (or {"error": "..."}\n)
  Client → Server: {"ping": 1}\n
  Server → Client: {"pong": 1}\n
"""

import sys
import os
import json
import base64
import io
import socket
import threading

# GPU + paths — must be set before any torch import
_gpu  = sys.argv[1] if len(sys.argv) > 1 else "1"
_port = int(sys.argv[2]) if len(sys.argv) > 2 else 18432
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
os.environ.setdefault("HF_HOME", "E:/project/Qwen3TTS")

import numpy as np
import soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
model = FasterQwen3TTS.from_pretrained(MODEL_ID, device="cuda")

# Single lock — model is not thread-safe
_lock = threading.Lock()


def _handle_client(conn: socket.socket) -> None:
    f = conn.makefile("r")
    try:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                req = json.loads(raw)

                if "ping" in req:
                    conn.sendall(b'{"pong":1}\n')
                    continue

                with _lock:
                    wavs, sr = model.generate_voice_clone(
                        req["text"], "Russian",
                        req["ref_audio"], req.get("ref_text", ""),
                        xvec_only=True,
                        non_streaming_mode=True,
                    )

                audio = np.array(wavs[0], dtype=np.float32)
                buf = io.BytesIO()
                sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
                wav_b64 = base64.b64encode(buf.getvalue()).decode()
                conn.sendall((json.dumps({"wav_b64": wav_b64, "sr": sr}) + "\n").encode())

            except Exception as e:
                try:
                    conn.sendall((json.dumps({"error": str(e)}) + "\n").encode())
                except Exception:
                    break
    except Exception:
        pass
    finally:
        try:
            f.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


# Signal parent process that model is loaded
print("READY", flush=True)

# Start TCP server
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", _port))
srv.listen(5)

while True:
    conn, _ = srv.accept()
    threading.Thread(target=_handle_client, args=(conn,), daemon=True).start()
