"""
Qwen3-TTS engine — client for the persistent qwen3_worker daemon.

On first start: spawns qwen3_worker.py as a DETACHED subprocess in qwen3tts_env,
waits for "READY" (~60s), then connects via TCP socket.

On subsequent server restarts: daemon is still running → connects instantly (<1s).
No model reload, no forrtl, no wait.
"""

import os
import socket
import base64
import asyncio
import subprocess
import json
import time
import threading

from app.config import load_config
from app.monitoring.logger import setup_logger

QWEN3_PYTHON = "E:/project/qwen3tts_env/Scripts/python.exe"
WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_worker.py")
DEFAULT_PORT   = 18432


class Qwen3TTSEngine:
    """
    ZeroShot TTS via persistent Qwen3 daemon process.
    First start: loads model (~60s). Restarts: instant socket reconnect.
    """

    output_sample_rate: int = 24000
    speed: float = 1.0  # speed handled via ffmpeg atempo in worker pool

    def __init__(self):
        self.logger = setup_logger(__name__)
        cfg = load_config()["models"]["tts"]

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        voice_sample = cfg.get("voice_sample", "voice_samples/Aleksandr.wav")
        self.ref_audio = os.path.join(project_root, voice_sample).replace("\\", "/")
        self.ref_text  = cfg.get("qwen3_ref_text", "")
        self.output_sample_rate = cfg.get("output_sample_rate", 24000)
        gpu_id         = cfg.get("gpu_id", 1)
        self._port     = cfg.get("daemon_port", DEFAULT_PORT)
        self._sock     = None
        self._sockfile = None

        if not os.path.exists(self.ref_audio):
            raise FileNotFoundError(f"voice_sample not found: {self.ref_audio}")

        # ── Try existing daemon first ──────────────────────────────────────
        if self._try_connect():
            self.logger.info(
                f"[Qwen3TTS] ⚡ Daemon already running on port {self._port} — instant connect!"
            )
            return

        # ── No daemon → start one ─────────────────────────────────────────
        self.logger.info(f"[Qwen3TTS] Starting daemon subprocess (GPU={gpu_id})")
        self.logger.info(f"[Qwen3TTS] Python: {QWEN3_PYTHON}")
        self.logger.info(f"[Qwen3TTS] ref_audio: {self.ref_audio}")

        sox_dir = (
            r"C:\Users\bogdan\AppData\Local\Microsoft\WinGet\Packages"
            r"\ChrisBagwell.SoX_Microsoft.Winget.Source_8wekyb3d8bbwe\sox-14.4.2"
        )
        env = os.environ.copy()
        if os.path.isdir(sox_dir):
            env["PATH"] = sox_dir + os.pathsep + env.get("PATH", "")

        # No stdout/stderr pipes — avoids Windows pipe deadlock and spawn-mode pipe issues.
        # Daemon readiness is detected by polling the TCP socket it binds after loading.
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        proc = subprocess.Popen(
            [QWEN3_PYTHON, WORKER_SCRIPT, str(gpu_id), str(self._port)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            creationflags=CREATE_NEW_PROCESS_GROUP,
        )

        self.logger.info("[Qwen3TTS] Polling for daemon TCP socket (first time ~60s)...")
        deadline = time.time() + 150
        while time.time() < deadline:
            if self._try_connect():
                break
            rc = proc.poll()
            if rc is not None:
                raise RuntimeError(f"Qwen3 daemon process crashed (rc={rc})")
            time.sleep(2)
        else:
            proc.kill()
            raise RuntimeError("Qwen3 daemon startup timeout (150s)")

        # Give the server socket a moment to bind
        for attempt in range(20):
            if self._try_connect():
                break
            time.sleep(0.3)
        else:
            raise RuntimeError(
                f"Daemon printed READY but socket on port {self._port} "
                "is not accepting connections"
            )

        # Close pipes — daemon communicates via socket from here on
        try:
            proc.stdout.close()
            proc.stderr.close()
        except Exception:
            pass

        self.logger.info("[Qwen3TTS] ✅ Daemon ready and connected via socket")

    # ── Socket helpers ────────────────────────────────────────────────────

    def _try_connect(self) -> bool:
        """Return True if daemon is alive and ping succeeds."""
        try:
            sock = socket.create_connection(("127.0.0.1", self._port), timeout=1.0)
            sock.sendall(b'{"ping":1}\n')
            resp = sock.recv(64).decode("utf-8", errors="replace")
            if '"pong"' in resp:
                sock.settimeout(None)          # blocking BEFORE makefile
                f = sock.makefile("r")         # makefile inherits blocking mode
                self._sock     = sock
                self._sockfile = f
                return True
            sock.close()
        except Exception:
            pass
        return False

    def _reconnect(self) -> None:
        """Close stale socket and reconnect (daemon is still running)."""
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        self._sock     = None
        self._sockfile = None

        for _ in range(10):
            if self._try_connect():
                self.logger.info("[Qwen3TTS] Reconnected to daemon")
                return
            time.sleep(0.5)
        raise RuntimeError("Qwen3 daemon not responding — may have crashed")

    # ── Public API ────────────────────────────────────────────────────────

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text. Returns WAV bytes."""
        self.logger.debug(f"[Qwen3TTS] Synthesizing {len(text)} chars")
        req = json.dumps({
            "text":      text,
            "ref_audio": self.ref_audio,
            "ref_text":  self.ref_text,
        })
        response_line = await asyncio.to_thread(self._send_request, req)
        result = json.loads(response_line)
        if "error" in result:
            raise RuntimeError(f"Qwen3 daemon error: {result['error']}")
        return base64.b64decode(result["wav_b64"])

    def _send_request(self, req_json: str) -> str:
        """Send one JSON request over the socket, return one JSON response line."""
        try:
            self._sock.sendall((req_json + "\n").encode())
            return self._sockfile.readline().strip()
        except Exception:
            self._reconnect()
            self._sock.sendall((req_json + "\n").encode())
            return self._sockfile.readline().strip()

    def __del__(self):
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
