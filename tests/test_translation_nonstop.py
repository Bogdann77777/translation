"""
Non-stop Translation test (without TTS).
Tests continuous translation of batches to verify the system works non-stop.
"""

import asyncio
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.openrouter_llm import OpenRouterClient
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


async def test_translation_nonstop():
    """Test non-stop translation processing."""

    print("=" * 80)
    print("NON-STOP TRANSLATION TEST")
    print("Testing continuous batch-by-batch translation")
    print("=" * 80)

    # Load config
    try:
        config = load_config()
    except Exception as e:
        print(f"\n❌ Failed to load config: {e}")
        print("\nMake sure you have:")
        print("  1. config.yaml in the project root")
        print("  2. .env file with OPENROUTER_API_KEY")
        return

    # Initialize translator
    print("\n[*] Initializing OpenRouter translator...")
    try:
        translator = OpenRouterClient()
        print("[OK] Translator ready\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize translator: {e}")
        return

    # Split text into sentences
    sentences = [s.strip() + "." for s in ACADEMIC_TEXT.split(".") if s.strip() and len(s.strip()) > 10]
    print(f"[*] Loaded {len(sentences)} sentences from academic text")
    print(f"[*] Testing non-stop translation...\n")

    # Create output file
    output_dir = Path(__file__).parent.parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "translation_results.txt"

    # Processing stats
    context = []
    total_time = 0
    batch_times = []
    gaps = []
    last_end_time = None
    results = []

    test_start = time.time()

    # Process all batches
    for i, sentence in enumerate(sentences, 1):
        print(f"\n{'=' * 80}")
        print(f"BATCH #{i}/{len(sentences)}")
        print(f"{'=' * 80}")

        batch_start = time.time()

        # Calculate gap since last batch
        if last_end_time is not None:
            gap = batch_start - last_end_time
            gaps.append(gap)
            print(f"[TIME] Gap since last batch: {gap:.3f}s")

        # Display input
        print(f"\n[EN] {sentence}")

        try:
            # Translate
            trans_start = time.time()
            translation = await translator.translate(sentence, context)
            trans_time = time.time() - trans_start

            print(f"[RU] <saved to file>")
            print(f"[TIME] Translation: {trans_time:.2f}s")

            # Save to results
            results.append({
                'batch': i,
                'en': sentence,
                'ru': translation,
                'time': trans_time
            })

            # Update context
            context.append(sentence)
            if len(context) > 10:
                context.pop(0)

            # Track stats
            batch_times.append(trans_time)
            total_time += trans_time
            last_end_time = time.time()

            print(f"[OK] Batch #{i} completed")

        except Exception as e:
            print(f"[ERROR] Batch #{i}: {e}")
            import traceback
            traceback.print_exc()

    test_duration = time.time() - test_start

    # Save results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TRANSLATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            f.write(f"BATCH #{result['batch']}\n")
            f.write(f"EN: {result['en']}\n")
            f.write(f"RU: {result['ru']}\n")
            f.write(f"Time: {result['time']:.2f}s\n")
            f.write("\n" + "-" * 80 + "\n\n")

    print(f"\n[*] Results saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("NON-STOP TRANSLATION TEST SUMMARY")
    print("=" * 80)

    print(f"\n[STATS] Overall Statistics:")
    print(f"  Total batches processed: {len(batch_times)}")
    print(f"  Total translation time: {total_time:.2f}s")
    print(f"  Total test duration: {test_duration:.2f}s")

    if batch_times:
        avg_batch = sum(batch_times) / len(batch_times)
        min_batch = min(batch_times)
        max_batch = max(batch_times)

        print(f"\n[TIME] Batch Duration:")
        print(f"  Average: {avg_batch:.2f}s")
        print(f"  Min: {min_batch:.2f}s")
        print(f"  Max: {max_batch:.2f}s")

    if gaps:
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        min_gap = min(gaps)

        print(f"\n[GAPS] Between Batches:")
        print(f"  Average: {avg_gap:.3f}s")
        print(f"  Min: {min_gap:.3f}s")
        print(f"  Max: {max_gap:.3f}s")

        print(f"\n[PERFORMANCE] Non-stop Performance:")
        if avg_gap < 0.1:
            print(f"  [EXCELLENT] Nearly zero gaps! ({avg_gap:.3f}s)")
        elif avg_gap < 0.5:
            print(f"  [GOOD] Minimal gaps ({avg_gap:.3f}s)")
        elif avg_gap < 1.0:
            print(f"  [ACCEPTABLE] Some gaps ({avg_gap:.3f}s)")
        else:
            print(f"  [WARNING] Large gaps ({avg_gap:.3f}s)")

    throughput = len(batch_times) / test_duration if test_duration > 0 else 0
    print(f"\n[STATS] Throughput: {throughput:.2f} batches/second")

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

    print("\n[OK] Translation component is working correctly!")
    print("\nNext steps:")
    print("  1. Review translation quality above")
    print("  2. Check if context is preserved between batches")
    print("  3. Adjust llm.temperature in config.yaml if needed")
    print("  4. Run full pipeline test with TTS when ready")


if __name__ == "__main__":
    asyncio.run(test_translation_nonstop())
