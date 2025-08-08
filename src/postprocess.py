import os
import io
import json
import time
import base64
import hashlib
import sqlite3
from typing import Any, Dict, Optional, Union

import numpy as np
import requests
import soundfile as sf
import torch

import httpx
from openai import OpenAI

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class PostProcessor:
    """Postprocessing utilities for ASR, summarization, and KV storage.

    - Transcription via local Whisper model
    - Summarization via OpenAI-compatible API (`/v1/chat/completions`)
    - KV store backed by SQLite, keyed by a deterministic fingerprint derived from the speaker embedding
    """

    def __init__(
        self,
        db_path: str = "voice_kv.sqlite",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        whisper_model: str = "base",
        chat_model: Optional[str] = None,
        timeout_seconds: int = 120,
        device: Optional[str] = None,
    ) -> None:
        self.db_path = db_path
        self.api_base = api_base or os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        print(os.getenv("OPENAI_API_KEY", ""))
        self.whisper_model = whisper_model or os.getenv("WHISPER_MODEL", "base")
        self.chat_model = chat_model or os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.timeout_seconds = timeout_seconds
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.api_base, self.api_key, self.chat_model)

        # Initialize Whisper model
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper not available. Install with: pip install openai-whisper")
        
        print(f"Loading Whisper model: {self.whisper_model}")
        self.whisper = whisper.load_model(self.whisper_model, device=self.device)
        print(f"Whisper model loaded on device: {self.device}")

        # Check if we have API key for summarization (optional)
        if not self.api_key:
            print("Warning: No API key provided. Summarization will be skipped.")
            print("Set OPENAI_API_KEY environment variable to enable summarization.")

        self._ensure_db()

    # --------------- Public API ---------------

    def process_and_store(
        self,
        audio_path: str,
        speaker_embedding: Union[torch.Tensor, np.ndarray, list],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Transcribe, summarize, and store results keyed by speaker fingerprint.

        Returns a dictionary containing the key and stored value fields.
        """
        fingerprint = self.compute_embedding_fingerprint(speaker_embedding)
        transcript_text = self.transcribe_audio(audio_path)
        print("Transcript: ", transcript_text)
        
        # Try to summarize if API key is available
        summary_text = ""
        if self.api_key:
            try:
                summary_text = self.summarize_text(transcript_text)
            except Exception as e:
                print(f"Summarization failed: {e}")
                summary_text = "Summarization unavailable"
        else:
            summary_text = "No API key provided for summarization"

        value: Dict[str, Any] = {
            "transcript": transcript_text,
            "summary": summary_text,
            "metadata": metadata or {},
            "audio_path": audio_path,
            "created_at": int(time.time()),
        }

        self.kv_put(fingerprint, value)
        return {"key": fingerprint, **value}

    def get_by_embedding(self, speaker_embedding: Union[torch.Tensor, np.ndarray, list]) -> Optional[Dict[str, Any]]:
        fingerprint = self.compute_embedding_fingerprint(speaker_embedding)
        return self.kv_get(fingerprint)

    # --------------- Fingerprint ---------------

    @staticmethod
    def compute_embedding_fingerprint(embedding: Union[torch.Tensor, np.ndarray, list]) -> str:
        """Compute a deterministic fingerprint from an embedding.

        Steps:
        - Convert to 1D float32 numpy array
        - L2 normalize
        - Quantize to 5 decimals to stabilize small float noise
        - SHA256 over raw bytes of the quantized vector
        - Return hex digest as key
        """
        if isinstance(embedding, torch.Tensor):
            emb = embedding.detach().cpu().float().numpy()
        elif isinstance(embedding, np.ndarray):
            emb = embedding.astype(np.float32)
        else:
            emb = np.asarray(embedding, dtype=np.float32)

        emb = emb.reshape(-1).astype(np.float32)
        norm = np.linalg.norm(emb) + 1e-12
        emb = emb / norm
        emb = np.round(emb, 5)
        digest = hashlib.sha256(emb.tobytes()).hexdigest()
        return digest

    # --------------- ASR (Local Whisper) ---------------

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using local Whisper model."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing: {audio_path}")
        result = self.whisper.transcribe(audio_path)
        return result["text"].strip()

    # --------------- Summarization ---------------

    def summarize_text(self, text: str, max_tokens: int = 256) -> str:
        """Summarize text via chat completions API."""
        if not self.api_key:
            raise ValueError("No API key provided for summarization")

        # url = self._join(self.api_base, "/v1/chat/completions")
        url = self.api_base
        system_prompt = (
            "You are a helpful assistant that writes concise summaries. "
            "Given a transcript, produce a short summary in 100 words."
        )
        user_prompt = text
        body = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        # resp = requests.post(url, headers=headers, json=body, timeout=self.timeout_seconds)

        endpoint = OpenAI(
            api_key=self.api_key,
            base_url=url,
            http_client=httpx.Client(
                verify=False  # important for company use
            )
        )
        completion = endpoint.chat.completions.create(**body)
        resp = completion.choices[0].message.content.strip()

        try:
            return resp
        except Exception:
            raise RuntimeError(f"Unexpected chat response schema: {resp}")

    # --------------- KV Store (SQLite) ---------------

    def _ensure_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
                """
            )
            conn.commit()

    def kv_put(self, key: str, value: Dict[str, Any]) -> None:
        record = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        created_at = int(time.time())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv_store(key, value, created_at) VALUES (?, ?, ?)",
                (key, record, created_at),
            )
            conn.commit()

    def kv_get(self, key: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row:
                return None
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return None

    # --------------- Helpers ---------------

    @staticmethod
    def _join(base: str, path: str) -> str:
        if base.endswith("/"):
            base = base[:-1]
        return f"{base}{path}"

    @staticmethod
    def _infer_mime(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".wav"}:  # default
            return "audio/wav"
        if ext in {".flac"}:
            return "audio/flac"
        if ext in {".mp3"}:
            return "audio/mpeg"
        if ext in {".m4a", ".aac"}:
            return "audio/aac"
        return "application/octet-stream"

    def _raise_for_status(self, resp):
        """Raise an exception for bad HTTP status codes."""
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}") 