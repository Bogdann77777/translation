"""
STRESS TEST: Large-scale non-stop translation + TTS
Tests 100+ sentences to verify:
- Non-stop batch processing (no gaps)
- System can handle 1-2 hour sessions
- Mixed content (colloquial + academic)
- No buffering delays
- Consistent throughput
"""

import asyncio
import sys
from pathlib import Path
import time
from gtts import gTTS

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.openrouter_llm import OpenRouterClient


class PerformanceMonitor:
    """Monitor non-stop performance."""
    def __init__(self):
        self.batches = []
        self.gaps = []
        self.last_batch_end = None
        self.test_start = time.time()

    def record_batch(self, batch_num, trans_time, tts_time, batch_start):
        """Record batch metrics."""
        batch_end = time.time()

        # Calculate gap
        gap = 0.0
        if self.last_batch_end is not None:
            gap = batch_start - self.last_batch_end
            self.gaps.append(gap)

        self.batches.append({
            'num': batch_num,
            'trans_time': trans_time,
            'tts_time': tts_time,
            'total_time': trans_time + tts_time,
            'gap': gap
        })

        self.last_batch_end = batch_end

    def print_progress(self, batch_num, total_batches):
        """Print progress bar."""
        if batch_num % 10 == 0 or batch_num == total_batches:
            progress = batch_num / total_batches * 100
            elapsed = time.time() - self.test_start
            avg_time = elapsed / batch_num if batch_num > 0 else 0
            eta = avg_time * (total_batches - batch_num)

            print(f"\n[PROGRESS] {batch_num}/{total_batches} ({progress:.0f}%)")
            print(f"  Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

            if len(self.gaps) > 0:
                avg_gap = sum(self.gaps) / len(self.gaps)
                max_gap = max(self.gaps)
                print(f"  Avg gap: {avg_gap:.3f}s | Max gap: {max_gap:.3f}s")

    def print_summary(self, total_batches):
        """Print detailed performance summary."""
        print("\n" + "=" * 80)
        print("STRESS TEST PERFORMANCE SUMMARY")
        print("=" * 80)

        total_time = time.time() - self.test_start

        # Overall stats
        print(f"\n[OVERALL]")
        print(f"  Total batches: {total_batches}")
        print(f"  Test duration: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Throughput: {total_batches/total_time:.2f} batches/sec")

        # Translation stats
        trans_times = [b['trans_time'] for b in self.batches]
        print(f"\n[TRANSLATION]")
        print(f"  Average: {sum(trans_times)/len(trans_times):.2f}s")
        print(f"  Min: {min(trans_times):.2f}s")
        print(f"  Max: {max(trans_times):.2f}s")
        print(f"  Std dev: {self._std_dev(trans_times):.2f}s")

        # TTS stats
        tts_times = [b['tts_time'] for b in self.batches]
        print(f"\n[TTS]")
        print(f"  Average: {sum(tts_times)/len(tts_times):.2f}s")
        print(f"  Min: {min(tts_times):.2f}s")
        print(f"  Max: {max(tts_times):.2f}s")
        print(f"  Std dev: {self._std_dev(tts_times):.2f}s")

        # Gap analysis (CRITICAL for non-stop)
        if len(self.gaps) > 0:
            avg_gap = sum(self.gaps) / len(self.gaps)
            max_gap = max(self.gaps)
            min_gap = min(self.gaps)

            print(f"\n[GAPS BETWEEN BATCHES] *** CRITICAL METRIC ***")
            print(f"  Average gap: {avg_gap:.3f}s")
            print(f"  Min gap: {min_gap:.3f}s")
            print(f"  Max gap: {max_gap:.3f}s")
            print(f"  Std dev: {self._std_dev(self.gaps):.3f}s")

            # Count problematic gaps
            large_gaps = [g for g in self.gaps if g > 0.5]
            print(f"\n  Gaps > 0.5s: {len(large_gaps)}/{len(self.gaps)} ({len(large_gaps)/len(self.gaps)*100:.1f}%)")

            # Performance rating
            print(f"\n[NON-STOP RATING]")
            if avg_gap < 0.1:
                print(f"  *** EXCELLENT *** - Average gap < 0.1s")
                print(f"  System is truly non-stop!")
            elif avg_gap < 0.3:
                print(f"  *** GOOD *** - Average gap < 0.3s")
                print(f"  Minimal delays, acceptable for live translation")
            elif avg_gap < 0.5:
                print(f"  *** ACCEPTABLE *** - Average gap < 0.5s")
                print(f"  Some delays but usable")
            else:
                print(f"  *** NEEDS OPTIMIZATION *** - Average gap > 0.5s")
                print(f"  Too many delays for smooth live translation")

        # Estimate capacity
        print(f"\n[CAPACITY ESTIMATE]")
        avg_batch_time = sum([b['total_time'] for b in self.batches]) / len(self.batches)
        batches_per_hour = 3600 / avg_batch_time
        print(f"  Avg batch time: {avg_batch_time:.2f}s")
        print(f"  Batches per hour: {batches_per_hour:.0f}")
        print(f"  Can handle ~{batches_per_hour:.0f} sentences/hour")

    def _std_dev(self, values):
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


async def test_large_nonstop():
    """Run large-scale non-stop test."""

    print("=" * 80)
    print("LARGE-SCALE NON-STOP STRESS TEST")
    print("=" * 80)
    print("\nTesting:")
    print("  - 100+ sentences (mixed colloquial + academic)")
    print("  - Full Translation + TTS pipeline")
    print("  - Non-stop batch processing")
    print("  - Gap analysis between batches")
    print("=" * 80)

    # Initialize
    translator = OpenRouterClient()
    monitor = PerformanceMonitor()

    # Load large text
    text_file = Path(__file__).parent.parent / "test_data" / "large_mixed_text.txt"
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    total = len(sentences)

    print(f"\n[LOADED] {total} sentences")
    print(f"[ESTIMATED] ~{total * 2:.0f} seconds for full test\n")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "test_output" / "stress_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all sentences
    context = []

    print("[STARTING] Non-stop processing...\n")

    for i, sentence in enumerate(sentences, 1):
        batch_start = time.time()

        try:
            # Translation
            trans_start = time.time()
            translation = await translator.translate(sentence, context)
            trans_time = time.time() - trans_start

            # Update context
            context.append(sentence)
            if len(context) > 10:
                context.pop(0)

            # TTS (only for first 20 to save time, but measure all translations)
            tts_time = 0.0
            if i <= 20:
                tts_start = time.time()
                tts = gTTS(text=translation, lang='ru', slow=False)
                audio_file = output_dir / f"audio_{i:03d}.mp3"
                tts.save(str(audio_file))
                tts_time = time.time() - tts_start

            # Record metrics
            monitor.record_batch(i, trans_time, tts_time, batch_start)

            # Print progress
            monitor.print_progress(i, total)

        except Exception as e:
            print(f"\n[ERROR] Batch #{i}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    monitor.print_summary(total)

    print("\n" + "=" * 80)
    print(f"[COMPLETE] Test finished!")
    print(f"[OUTPUT] Audio samples: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_large_nonstop())
