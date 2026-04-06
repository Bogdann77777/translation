"""
Adapter for LocalAgreement-2 algorithm to use our existing WhisperModel.

Wraps faster_whisper.WhisperModel with the interface expected by
SmartStreamProcessor (LocalAgreement-2 port from whisper_streaming).
"""

import numpy as np


class SmartWhisperASR:
    """
    Wraps our existing faster_whisper.WhisperModel with the ASR interface
    expected by the LocalAgreement-2 online processor.

    Key difference from FasterWhisperASR in whisper_streaming:
    - We REUSE the already-loaded model (no double VRAM usage)
    - sep = "" because faster-whisper includes spaces within word tokens
    """

    sep = " "  # normalize: strip words in ts_words(), join with space

    def __init__(self, model, language: str = "en"):
        """
        Args:
            model: Existing faster_whisper.WhisperModel instance
            language: Language hint (None = auto-detect, keeps Russian filter working)
        """
        self.model = model
        self.original_language = language

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> list:
        """
        Run Whisper transcription with word timestamps.

        Called by LocalAgreement processor every min_chunk_size seconds.
        Returns materialised list of segment objects.

        Args:
            audio: float32 numpy array at 16kHz
            init_prompt: Context from previously committed text

        Returns:
            list of faster_whisper Segment objects (with .words)
        """
        segments, info = self.model.transcribe(
            audio,
            language="en",
            initial_prompt=init_prompt or None,
            beam_size=1,
            word_timestamps=True,               # REQUIRED for LocalAgreement word-level alignment
            condition_on_previous_text=False,   # prevents repetition loops
            no_speech_threshold=0.45,
            compression_ratio_threshold=2.2,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=700
            )
        )
        return list(segments)  # Materialise the generator

    def ts_words(self, segments: list) -> list:
        """
        Convert segments to (start, end, word) tuples for LocalAgreement.

        Filters out high no-speech probability segments (silence/noise).

        Returns:
            list of (start_sec, end_sec, word_str) tuples
        """
        o = []
        for segment in segments:
            if getattr(segment, 'no_speech_prob', 0) > 0.45:
                continue
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    t = (word.start, word.end, word.word.strip())
                    o.append(t)
        return o

    def segments_end_ts(self, segments: list) -> list:
        """Return end timestamps of all segments (for buffer trimming)."""
        return [s.end for s in segments if s]

    def use_vad(self):
        return True
