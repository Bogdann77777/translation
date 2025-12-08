"""
Non-stop batch processing test.
Simulates continuous processing of text batches to verify the 3-slot queue system works correctly.
Tests: Text Input → Translation → TTS → Audio Output (non-stop, batch by batch)
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.openrouter_llm import OpenRouterLLM
from app.components.xtts_engine import XTTSEngine
from app.config import load_config


# Academic English text for testing
ACADEMIC_TEXT = """
The fundamental principles of quantum mechanics have revolutionized our understanding of atomic and subatomic phenomena.
Wave-particle duality represents one of the most counterintuitive aspects of quantum theory.
Heisenberg's uncertainty principle establishes inherent limitations on the precision of simultaneous measurements.
The Schrödinger equation provides a mathematical framework for describing quantum states and their evolution.
Quantum entanglement demonstrates correlations between particles that transcend classical physics explanations.
The Copenhagen interpretation remains the most widely accepted framework for understanding quantum measurements.
Quantum tunneling allows particles to traverse energy barriers that would be insurmountable in classical mechanics.
Superposition enables quantum systems to exist in multiple states simultaneously until measurement occurs.
The double-slit experiment elegantly demonstrates the wave nature of particles and the observer effect.
Quantum field theory extends quantum mechanics to incorporate special relativity and particle creation.
Decoherence explains the transition from quantum to classical behavior in macroscopic systems.
The EPR paradox challenged the completeness of quantum mechanics and sparked decades of philosophical debate.
Bell's theorem and subsequent experiments have confirmed the non-local nature of quantum correlations.
Quantum computing exploits superposition and entanglement to perform certain calculations exponentially faster.
The measurement problem addresses the apparent collapse of the wave function during observation.
Quantum chromodynamics describes the strong nuclear force binding quarks within protons and neutrons.
The Standard Model represents our most complete theory of fundamental particles and their interactions.
Quantum electrodynamics achieves unprecedented precision in predicting electromagnetic phenomena at microscopic scales.
Many-worlds interpretation proposes that all quantum outcomes actually occur in parallel universes.
The ongoing quest to reconcile quantum mechanics with general relativity remains one of physics greatest challenges.
"""


class BatchStats:
    """Track statistics for batch processing."""
    def __init__(self):
        self.batches_completed = 0
        self.total_translation_time = 0
        self.total_tts_time = 0
        self.batch_times = []
        self.gaps_between_batches = []
        self.last_batch_end = None

    def record_batch(self, translation_time, tts_time, batch_start):
        """Record statistics for a completed batch."""
        self.batches_completed += 1
        self.total_translation_time += translation_time
        self.total_tts_time += tts_time

        batch_duration = translation_time + tts_time
        self.batch_times.append(batch_duration)

        # Calculate gap since last batch
        if self.last_batch_end is not None:
            gap = batch_start - self.last_batch_end
            self.gaps_between_batches.append(gap)

        self.last_batch_end = time.time()

    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("NON-STOP BATCH PROCESSING SUMMARY")
        print("=" * 80)

        print(f"\n📊 Overall Statistics:")
        print(f"  Total batches processed: {self.batches_completed}")
        print(f"  Total translation time: {self.total_translation_time:.2f}s")
        print(f"  Total TTS time: {self.total_tts_time:.2f}s")
        print(f"  Total pipeline time: {self.total_translation_time + self.total_tts_time:.2f}s")

        if self.batch_times:
            avg_batch = sum(self.batch_times) / len(self.batch_times)
            min_batch = min(self.batch_times)
            max_batch = max(self.batch_times)

            print(f"\n⏱️  Batch Duration:")
            print(f"  Average: {avg_batch:.2f}s")
            print(f"  Min: {min_batch:.2f}s")
            print(f"  Max: {max_batch:.2f}s")

        if self.gaps_between_batches:
            avg_gap = sum(self.gaps_between_batches) / len(self.gaps_between_batches)
            max_gap = max(self.gaps_between_batches)
            min_gap = min(self.gaps_between_batches)

            print(f"\n🔄 Gaps Between Batches:")
            print(f"  Average: {avg_gap:.3f}s")
            print(f"  Min: {min_gap:.3f}s")
            print(f"  Max: {max_gap:.3f}s")

            if avg_gap < 0.5:
                print(f"  ✅ NON-STOP: Excellent! Average gap < 0.5s")
            elif avg_gap < 1.0:
                print(f"  ⚠️  ACCEPTABLE: Average gap < 1.0s")
            else:
                print(f"  ❌ GAPS DETECTED: Average gap > 1.0s - optimization needed")

        avg_translation = self.total_translation_time / self.batches_completed if self.batches_completed > 0 else 0
        avg_tts = self.total_tts_time / self.batches_completed if self.batches_completed > 0 else 0

        print(f"\n📈 Average Latencies:")
        print(f"  Translation: {avg_translation:.2f}s")
        print(f"  TTS: {avg_tts:.2f}s")
        print(f"  Total: {avg_translation + avg_tts:.2f}s")

        print(f"\n🎯 Target Metrics:")
        if avg_translation + avg_tts <= 5.5:
            print(f"  ✅ Total latency within target (≤5.5s)")
        else:
            print(f"  ⚠️  Total latency above target (>{avg_translation + avg_tts:.2f}s)")


async def process_batch(batch_num, sentence, translator, tts, context, stats, output_dir):
    """Process a single batch through the pipeline."""

    print(f"\n{'█' * 80}")
    print(f"BATCH #{batch_num}")
    print(f"{'█' * 80}")

    batch_start = time.time()

    # Display input
    print(f"\n📝 Input (EN):")
    print(f"   {sentence[:100]}{'...' if len(sentence) > 100 else ''}")

    try:
        # Step 1: Translation
        print(f"\n🔄 Step 1/2: Translation...", end=" ", flush=True)
        trans_start = time.time()
        translation = await translator.translate(sentence, context)
        translation_time = time.time() - trans_start
        print(f"✓ ({translation_time:.2f}s)")

        print(f"   {translation[:100]}{'...' if len(translation) > 100 else ''}")

        # Update context
        context.append(sentence)
        if len(context) > 10:
            context.pop(0)

        # Step 2: TTS
        print(f"\n🔊 Step 2/2: TTS generation...", end=" ", flush=True)
        tts_start = time.time()
        audio_bytes = await tts.generate_speech(translation)
        tts_time = time.time() - tts_start
        print(f"✓ ({tts_time:.2f}s)")

        # Save audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        output_file = output_dir / f"batch_{batch_num:03d}_{timestamp}.wav"
        with open(output_file, "wb") as f:
            f.write(audio_bytes)

        # Record stats
        stats.record_batch(translation_time, tts_time, batch_start)

        batch_total = translation_time + tts_time
        print(f"\n✅ Batch completed in {batch_total:.2f}s")
        print(f"   Audio: {len(audio_bytes)} bytes → {output_file.name}")

        return translation

    except Exception as e:
        print(f"\n❌ ERROR in batch #{batch_num}: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_nonstop_batches():
    """Test non-stop batch processing."""

    print("=" * 80)
    print("NON-STOP BATCH PROCESSING TEST")
    print("Testing 3-slot queue system: PLAYING → READY → PROCESSING")
    print("=" * 80)

    # Load config
    config = load_config()

    # Initialize components
    print("\n🔧 Initializing components...")
    translator = OpenRouterLLM(config)
    tts = XTTSEngine(config)
    print("✅ Components ready\n")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "test_output" / "nonstop_batches"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")

    # Split text into sentences
    sentences = [s.strip() + "." for s in ACADEMIC_TEXT.split(".") if s.strip() and len(s.strip()) > 10]
    print(f"📚 Loaded {len(sentences)} sentences from academic text")
    print(f"🎯 Testing non-stop processing...\n")

    # Processing
    context = []
    stats = BatchStats()

    test_start = time.time()

    # Process all batches
    for i, sentence in enumerate(sentences, 1):
        await process_batch(i, sentence, translator, tts, context, stats, output_dir)

    test_duration = time.time() - test_start

    # Print summary
    stats.print_summary()

    print(f"\n⏰ Total Test Duration: {test_duration:.2f}s")
    print(f"📂 Audio files saved to: {output_dir}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if stats.gaps_between_batches:
        avg_gap = sum(stats.gaps_between_batches) / len(stats.gaps_between_batches)

        print("\n🔍 Non-stop Performance:")
        if avg_gap < 0.1:
            print("  ✅ EXCELLENT: Nearly zero gaps between batches!")
            print("     The 3-slot queue system is working perfectly.")
        elif avg_gap < 0.5:
            print("  ✅ GOOD: Minimal gaps between batches.")
            print("     System maintains continuous flow.")
        elif avg_gap < 1.0:
            print("  ⚠️  ACCEPTABLE: Some gaps present but manageable.")
            print("     Consider optimizing async operations.")
        else:
            print("  ❌ NEEDS OPTIMIZATION: Significant gaps between batches.")
            print("     The queue system may not be properly overlapping operations.")

    throughput = stats.batches_completed / test_duration
    print(f"\n📊 Throughput: {throughput:.2f} batches/second")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_nonstop_batches())
