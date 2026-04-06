"""
Smart stream processor using LocalAgreement-2 algorithm.

Ported from https://github.com/ufal/whisper_streaming
Paper: "Turning Whisper into Real-Time Transcription System" (arXiv:2307.14743)

WHY THIS IS BETTER THAN VAD-BASED CHUNKING:
- VAD detects silence/speech but NOT sentence boundaries
- VAD cuts audio mid-sentence on breathing pauses, commas, short hesitations
- Whisper's internal language model KNOWS where sentences end (it predicts punctuation)
- LocalAgreement confirms text only when two consecutive Whisper runs agree
  (= the text won't change as more audio arrives = semantically stable)

ALGORITHM:
1. Accumulate rolling audio buffer (up to buffer_trimming_sec = 15s)
2. Every min_chunk_size seconds of NEW audio, run Whisper on entire buffer
3. Compare word sequence with PREVIOUS run (LocalAgreement):
   - Find longest common prefix of consecutive word sequences
   - Commit words appearing in BOTH runs (stable, won't change)
4. Trim buffer at last committed segment boundary
5. Emit committed text to translation pipeline

RESULT: Each emitted chunk ends at a natural Whisper segment boundary
(= complete thought, complete sentence, confirmed by two consecutive runs).
"""

import asyncio
import numpy as np
from app.monitoring.logger import setup_logger


class HypothesisBuffer:
    """
    Buffer for LocalAgreement-2 hypothesis tracking.

    Tracks words from consecutive Whisper runs and finds the
    longest stable prefix (words that appeared in both consecutive runs).

    Ported from whisper_streaming.whisper_online.HypothesisBuffer.
    """

    def __init__(self):
        self.commited_in_buffer = []    # (beg, end, word) confirmed in buffer tracking
        self.buffer = []                # pending words from PREVIOUS run
        self.new = []                   # words from CURRENT run (before LocalAgreement)
        self.last_commited_time = 0
        self.last_commited_word = None

    def insert(self, new, offset):
        """
        Insert new word hypothesis from current Whisper run.

        Args:
            new: list of (start, end, word) tuples (buffer-relative timestamps)
            offset: buffer_time_offset (absolute session time of buffer start)
        """
        # Convert buffer-relative timestamps to absolute session timestamps
        new = [(a + offset, b + offset, t) for a, b, t in new]

        # Filter words before last committed time (we already committed those)
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        # Handle overlap: if new run starts near where we left off,
        # check if it repeats the last few committed words (and skip them)
        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1])
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            for _ in range(i):
                                self.new.pop(0)
                            break

    def flush(self):
        """
        Commit words that appear in BOTH consecutive Whisper runs (LocalAgreement).

        LocalAgreement principle: a word is "stable" if the current run
        and the previous run both output the same word at this position.
        Only stable words are committed and sent to translation.

        Returns:
            list of committed (start, end, word) tuples
        """
        commit = []
        while self.new:
            na, nb, nt = self.new[0]
            if len(self.buffer) == 0:
                break
            if nt == self.buffer[0][2]:
                # Same word in both runs → COMMIT
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                # Divergence → stop committing (rest might change)
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        """Remove committed words before absolute 'time' from tracking buffer."""
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        """Return all words still in the pending (not yet committed) buffer."""
        return self.buffer


class SmartStreamProcessor:
    """
    Stream processor using LocalAgreement-2 for semantic chunking.

    Replaces VAD-based time cuts with Whisper-guided semantic boundaries.
    Each emitted chunk is a complete, stable thought confirmed by Whisper's
    own language model across two consecutive transcription runs.

    INTEGRATION:
    - Receives 100ms audio chunks from WebSocket (same as other processors)
    - Internally accumulates audio and runs Whisper every min_chunk_size seconds
    - When text is committed (stable), calls batch_queue.add_text_batch()
    - BatchQueue handles Translation + TTS (Whisper step is SKIPPED)
    """

    SAMPLE_RATE = 16000

    def __init__(
        self,
        batch_queue,
        whisper_model,
        min_chunk_size: float = 1.0,
        buffer_trimming_sec: float = 15.0
    ):
        """
        Args:
            batch_queue: BatchQueue to send committed text to
            whisper_model: Existing loaded WhisperModel (reuse, no double VRAM)
            min_chunk_size: Seconds of NEW audio before re-running Whisper (default: 1.0s)
                            Lower = more responsive, higher = more context per run
            buffer_trimming_sec: Max buffer before forced trim at segment boundary (default: 15s)
        """
        from app.components.smart_whisper_asr import SmartWhisperASR

        self.asr = SmartWhisperASR(whisper_model)
        self.batch_queue = batch_queue
        self.min_chunk_size = min_chunk_size
        self.buffer_trimming_sec = buffer_trimming_sec
        self.logger = setup_logger(__name__)

        # Audio state
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0.0           # Absolute session time of buffer start
        self.new_audio_samples = 0              # New samples since last Whisper run

        # LocalAgreement state
        self.transcript_buffer = HypothesisBuffer()
        self.commited = []                      # All committed (start, end, word) tuples

        # Concurrency control
        self.lock = asyncio.Lock()              # Protects audio_buffer and LocalAgreement state
        self._is_processing = False             # Prevents concurrent Whisper runs

        # Diagnostics counters
        self._chunk_count = 0                   # Total audio chunks received
        self._whisper_run_count = 0             # Total Whisper runs executed
        self._session_start = None              # Set on first chunk

        # Sentence accumulation buffer: collect committed words until complete sentence
        # Prevents "Spain" / "that" / "of course" from becoming individual TTS chunks
        self._pending_text = ""                 # Accumulated committed words not yet sent to TTS
        self._pending_last_update = 0.0         # Timestamp of last update (for timeout flush)
        self._min_sentence_chars = 40           # Min chars before considering a flush
        self._sentence_timeout_sec = 1.5        # Flush after silence IF >= _min_timeout_chars
        self._min_timeout_chars = 10            # Min chars to flush at short timeout (1.5s)
        self._long_timeout_sec = 0.8            # Force flush even short text after this long (was 2.0)
        self._last_flushed_words = []           # Tail of last flushed chunk (for cross-flush dedup)

        self.logger.info(
            f"SmartStreamProcessor initialized: "
            f"min_chunk={min_chunk_size}s, buffer_trim={buffer_trimming_sec}s "
            f"(LocalAgreement-2, sentence buffering, no VAD cuts)"
        )

    async def process_chunk(self, audio_bytes: bytes) -> None:
        """
        Process incoming 100ms audio chunk from WebSocket.

        Accumulates audio in buffer. When enough new audio accumulated,
        triggers a Whisper run + LocalAgreement. Committed text is sent
        to batch_queue for translation + TTS.
        """
        import time as _time2
        audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        self._chunk_count += 1
        if self._session_start is None:
            self._session_start = _time2.time()
        if self._chunk_count % 100 == 0:
            elapsed = _time2.time() - self._session_start
            self.logger.info(
                f"📊 SmartProcessor alive: {self._chunk_count} chunks received, "
                f"{self._whisper_run_count} Whisper runs, "
                f"session {elapsed:.0f}s, "
                f"_is_processing={self._is_processing}, "
                f"buffer={len(self.audio_buffer)/self.SAMPLE_RATE:.1f}s, "
                f"pending_text='{self._pending_text[:40] if self._pending_text else ''}'"
            )

        should_process = False
        audio_snapshot = None
        offset_snapshot = None

        async with self.lock:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_float])
            self.new_audio_samples += len(audio_float)

            new_audio_secs = self.new_audio_samples / self.SAMPLE_RATE
            if not self._is_processing and new_audio_secs >= self.min_chunk_size:
                should_process = True
                self._is_processing = True
                self.new_audio_samples = 0
                audio_snapshot = self.audio_buffer.copy()
                offset_snapshot = self.buffer_time_offset

        if should_process:
            # Cap snapshot to buffer_trimming_sec — Whisper never runs on >15s of audio
            # Prevents the buffer from causing 20-40s Whisper runs that stall the pipeline
            max_samples = int(self.buffer_trimming_sec * self.SAMPLE_RATE)
            if len(audio_snapshot) > max_samples:
                excess = len(audio_snapshot) - max_samples
                audio_snapshot = audio_snapshot[excess:]
                offset_snapshot += excess / self.SAMPLE_RATE
                self.logger.warning(
                    f"⚡ Snapshot capped to {self.buffer_trimming_sec:.0f}s "
                    f"(buffer was {(excess + max_samples) / self.SAMPLE_RATE:.0f}s)"
                )

            import time as _time3
            self._whisper_run_count += 1
            run_num = self._whisper_run_count
            self.logger.debug(f"🎙️ Whisper run #{run_num} START (buffer {len(audio_snapshot)/self.SAMPLE_RATE:.1f}s)")
            whisper_start = _time3.time()
            try:
                committed_text = await self._run_process_iter(audio_snapshot, offset_snapshot)
                whisper_elapsed = _time3.time() - whisper_start
                self.logger.debug(f"🎙️ Whisper run #{run_num} DONE in {whisper_elapsed:.2f}s → committed: '{committed_text.strip() if committed_text else ''}'")
            except Exception as e:
                import traceback as _tb
                self.logger.error(f"🔴 Whisper run #{run_num} EXCEPTION: {type(e).__name__}: {e}\n{''.join(_tb.format_exc())}")
                committed_text = ""
            finally:
                async with self.lock:
                    self._is_processing = False

            if committed_text and committed_text.strip():
                self.logger.info(f"🧠 LocalAgreement COMMITTED: '{committed_text.strip()}'")
                await self._accumulate_and_maybe_flush(committed_text.strip())

        # Two-tier timeout flush:
        # - Short timeout (1.5s): only flush if accumulated text >= _min_timeout_chars (25)
        #   Prevents micro-chunks ("we go", "cooks overnight") from going to LLM alone
        # - Long timeout (4.0s): force flush regardless of length (safety net)
        import time as _time
        if self._pending_text and self._pending_last_update > 0:
            waited = _time.time() - self._pending_last_update
            pending_len = len(self._pending_text.strip())
            if waited > self._long_timeout_sec:
                self.logger.info(f"⏱️ Long timeout ({waited:.1f}s) flush: '{self._pending_text[:60]}'")
                await self._flush_pending("long_timeout")
            elif waited > self._sentence_timeout_sec and pending_len >= self._min_timeout_chars:
                self.logger.info(f"⏱️ Timeout flush ({waited:.1f}s, {pending_len} chars): '{self._pending_text[:60]}'")
                await self._flush_pending("timeout")

    async def _accumulate_and_maybe_flush(self, new_text: str) -> None:
        """
        Accumulate committed words until a complete sentence is formed.

        Flush to translation+TTS when:
        1. Text ends with sentence-terminal punctuation (. ? ! ...)
        2. Accumulated text exceeds _min_sentence_chars and new_text ends sentence-ish
        3. Timeout (_sentence_timeout_sec) — safety net for sentences without punctuation

        This prevents 1-3 word commits like "Spain", "that", "of course"
        from each becoming separate TTS chunks (which sound choppy and flood the queue).
        """
        import time as _time

        # Deduplication: LocalAgreement can re-commit words already in _pending_text
        # when Whisper assigns slightly different timestamps across consecutive runs
        # on the same rolling audio buffer.
        # Fix: check if new_text starts with words already at the END of _pending_text
        # and remove the overlapping prefix before accumulating.
        if self._pending_text and new_text.strip():
            pending_words = self._pending_text.split()
            new_words = new_text.strip().split()
            max_overlap = min(len(pending_words), len(new_words), 30)
            removed = 0
            for overlap_size in range(max_overlap, 3, -1):  # require at least 4-word match
                pw = [w.lower().strip(".,!?") for w in pending_words[-overlap_size:]]
                nw = [w.lower().strip(".,!?") for w in new_words[:overlap_size]]
                if pw == nw:
                    new_words = new_words[overlap_size:]
                    removed = overlap_size
                    break
            if removed:
                self.logger.info(f"🔄 Dedup: removed {removed} re-committed words from '{new_text.strip()[:60]}'")
                new_text = " ".join(new_words)
            if not new_text.strip():
                return  # Entire commit was a duplicate — skip

        # Cross-flush dedup: detect when LocalAgreement re-commits words from the PREVIOUS chunk.
        # Happens when a chunk is flushed mid-sentence (long_clause/timeout) and the next Whisper
        # run re-confirms the tail of that chunk as "stable", producing a duplicate mid-text.
        # Example: chunk5="...The world was waking" → chunk6="up unbothered. The world was waking up..."
        #          → truncate at the repeat → chunk6="up unbothered."
        if self._last_flushed_words and not self._pending_text and new_text.strip():
            new_words = new_text.strip().split()
            found_at = -1
            for start_pos in range(1, len(new_words)):
                remaining = new_words[start_pos:]
                for overlap_size in range(min(len(self._last_flushed_words), len(remaining), 10), 3, -1):
                    pw = [w.lower().strip(".,!?") for w in self._last_flushed_words[-overlap_size:]]
                    nw = [w.lower().strip(".,!?") for w in remaining[:overlap_size]]
                    if pw == nw:
                        found_at = start_pos
                        break
                if found_at >= 0:
                    break
            if found_at > 0:
                removed = " ".join(new_words[found_at:])
                new_text = " ".join(new_words[:found_at])
                self.logger.info(f"🔄 Cross-flush dedup: truncated repeat at pos {found_at}: '{removed[:60]}'")
            if not new_text.strip():
                return

        # Append to pending buffer
        if self._pending_text:
            self._pending_text = self._pending_text + self.asr.sep + new_text
        else:
            self._pending_text = new_text
        self._pending_last_update = _time.time()

        pending = self._pending_text.strip()
        total_chars = len(pending)

        # Check for sentence-terminal punctuation
        ends_sentence = pending.rstrip().endswith(('.', '?', '!', '...', '。', '！', '？'))

        should_flush = False
        flush_reason = ""

        if ends_sentence:
            should_flush = True
            flush_reason = "sentence_end"
        elif total_chars >= self._min_sentence_chars and any(c in new_text for c in '.?!,;'):
            # Long enough AND ends with any pause marker — good cut point
            should_flush = True
            flush_reason = f"long_clause ({total_chars} chars)"
        elif total_chars >= 150:
            # Safety: never accumulate more than ~150 chars regardless
            should_flush = True
            flush_reason = f"max_length ({total_chars} chars)"

        if should_flush:
            await self._flush_pending(flush_reason)

    async def _flush_pending(self, reason: str) -> None:
        """Flush accumulated pending text to translation+TTS pipeline."""
        if not self._pending_text or not self._pending_text.strip():
            return
        text = self._pending_text.strip()
        self._pending_text = ""
        self._pending_last_update = 0.0
        # Save tail for cross-flush dedup (next commit after this flush will check against it)
        words = text.split()
        self._last_flushed_words = words[-30:]
        self.logger.info(f"🧠 FLUSHING ({reason}): '{text}'")
        await self.batch_queue.add_text_batch(text)

    async def _run_process_iter(self, audio_snapshot: np.ndarray, offset_snapshot: float) -> str:
        """
        Run one LocalAgreement iteration:
        1. Build context prompt from previously committed text
        2. Run Whisper on entire buffer snapshot (in thread pool)
        3. Apply LocalAgreement: compare with previous run, commit stable words
        4. Trim buffer if too large

        Returns:
            Committed text string (may be empty if no stable words yet)
        """
        # Build prompt from recent committed text (context for better transcription)
        async with self.lock:
            prompt = self._build_prompt(offset_snapshot)

        # Run Whisper in thread pool (blocking call, ~0.5-1s)
        try:
            segments = await asyncio.to_thread(self.asr.transcribe, audio_snapshot, prompt)
        except Exception as e:
            self.logger.error(f"Whisper error in LocalAgreement: {e}")
            return ""

        # Apply LocalAgreement (must be atomic)
        async with self.lock:
            tsw = self.asr.ts_words(segments)

            # Insert new hypothesis and find stable prefix
            self.transcript_buffer.insert(tsw, offset_snapshot)
            committed_words = self.transcript_buffer.flush()
            self.commited.extend(committed_words)

            # Build committed text (faster-whisper sep="" means spaces are in the words)
            committed_text = self.asr.sep.join(w for _, _, w in committed_words)

            # Trim buffer at last committed word boundary (safe for LocalAgreement).
            # Force-trimming at arbitrary points breaks LocalAgreement because consecutive
            # Whisper runs would see shifted audio and fail to find a stable prefix.
            # Snapshot cap above ensures Whisper always runs fast regardless of buffer size.
            buffer_secs = len(self.audio_buffer) / self.SAMPLE_RATE
            if buffer_secs > self.buffer_trimming_sec:
                last_committed = self.transcript_buffer.last_commited_time
                if last_committed > self.buffer_time_offset + 1.0:
                    self._chunk_at(last_committed)
                    new_secs = len(self.audio_buffer) / self.SAMPLE_RATE
                    self.logger.debug(
                        f"🔪 Buffer trimmed at commit boundary: {buffer_secs:.0f}s → {new_secs:.0f}s"
                    )

        return committed_text

    def _build_prompt(self, offset_snapshot: float) -> str:
        """
        Build Whisper prompt from recently committed text.

        Provides context to Whisper for better continuation transcription.
        Uses words committed before the current buffer window.
        Max 200 characters.
        """
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > offset_snapshot:
            k -= 1
        p = [t for _, _, t in self.commited[:k]]

        prompt_words = []
        length = 0
        while p and length < 200:
            x = p.pop(-1)
            length += len(x) + 1
            prompt_words.append(x)
        return self.asr.sep.join(reversed(prompt_words))

    def _chunk_completed_segment(self, segments: list) -> None:
        """
        Trim audio buffer at the second-to-last segment boundary.

        Keeps the last segment in the buffer (it may still be in progress),
        trims everything before the second-to-last completed segment.

        This prevents the buffer from growing unboundedly during long speeches.
        Ported from whisper_streaming.OnlineASRProcessor.chunk_completed_segment.
        """
        if not self.commited:
            return

        ends = self.asr.segments_end_ts(segments)
        t = self.commited[-1][1]  # Absolute time of last committed word

        if len(ends) > 1:
            # Use second-to-last segment end (buffer-relative) as cut point
            e = ends[-2] + self.buffer_time_offset  # Convert to absolute time
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset

            if e <= t:
                self._chunk_at(e)
                self.logger.debug(f"Buffer trimmed at segment boundary: {e:.1f}s")

    def _chunk_at(self, time: float) -> None:
        """
        Trim audio buffer and advance buffer_time_offset.

        Args:
            time: Absolute session timestamp to trim at
        """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        if cut_seconds > 0:
            cut_samples = int(cut_seconds * self.SAMPLE_RATE)
            self.audio_buffer = self.audio_buffer[cut_samples:]
            self.buffer_time_offset = time
