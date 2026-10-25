"""Math verification with a time limit.

SymPy can run for minutes on some expressions (for example expanding (a+b+c+d+e+f)^25), and a Python
thread cannot be stopped. Verification therefore runs in one long-lived helper process
(`src.verification.worker`) that is killed and restarted when a check takes too long.
"""

import json
import logging
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TimeLimitedVerifier:
    """Calls `verify_math_solution` in a helper process; a check that is late, or arrives while the
    helper is busy or still starting, is skipped (returns None) so an answer is never held up."""

    def __init__(self, timeout: float = 3.0, busy_wait: float = 0.5):
        self.timeout = timeout
        self.busy_wait = busy_wait
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._replies: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._ready = False

    def start(self) -> None:
        """Launch the helper now so it is warm by the first answer."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._spawn()

    def verify(self, answer: str) -> Optional[Dict[str, Any]]:
        if not answer:
            return None
        if not self._lock.acquire(timeout=self.busy_wait):
            return None
        try:
            if self._process is None or self._process.poll() is not None:
                self._spawn()
                return None
            if not self._ready and not self._take_ready():
                return None
            self._process.stdin.write(json.dumps({"answer": answer}) + "\n")
            self._process.stdin.flush()
            try:
                reply = self._replies.get(timeout=self.timeout)
            except queue.Empty:
                logger.warning("Math check took longer than %.1f s; restarting the verifier", self.timeout)
                self._spawn()
                return None
            return reply.get("result")
        except (OSError, ValueError):  # the helper died mid-request
            self._spawn()
            return None
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            self._kill()

    def _take_ready(self) -> bool:
        try:
            self._ready = bool(self._replies.get_nowait().get("ready"))
        except queue.Empty:
            return False
        return self._ready

    def _spawn(self) -> None:
        self._kill()
        self._replies = queue.Queue()
        self._ready = False
        self._process = subprocess.Popen(
            [sys.executable, "-m", "src.verification.worker"],
            cwd=str(_PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        threading.Thread(
            target=self._read_replies, args=(self._process, self._replies), name="verifier-replies", daemon=True
        ).start()

    @staticmethod
    def _read_replies(process: subprocess.Popen, replies: "queue.Queue[Dict[str, Any]]") -> None:
        for line in process.stdout:
            try:
                replies.put(json.loads(line))
            except ValueError:
                continue

    def _kill(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None
