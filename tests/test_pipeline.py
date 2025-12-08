"""
Full pipeline test: Text → Translation → TTS → Audio
Tests the complete flow without audio streaming input.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.openrouter_llm import OpenRouterLLM
from app.components.xtts_engine import XTTSEngine
from app.config import load_config


async def test_full_pipeline():
    """Test complete translation + TTS pipeline."""
    print("=" * 80)
    print("FULL PIPELINE TEST: Translation → TTS")
    print("=" * 80)

    # Load config
    config = load_config()

    # Initialize components
    print("\nInitializing components...")
    translator = OpenRouterLLM(config)
    tts = XTTSEngine(config)
    print("✓ Components initialized\n")

    # Load sample text
    sample_file = Path(__file__).parent.parent / "test_data" / "sample_text.txt"
    with open(sample_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into sentences
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    print(f"Loaded {len(sentences)} sentences\n")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "test_output"
    output_dir.mkdir(exist_ok=True)

    # Process each sentence through full pipeline
    context = []
    total_translation_time = 0
    total_tts_time = 0

    for i, sentence in enumerate(sentences, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing {i}/{len(sentences)}")
        print(f"{'=' * 80}")
        print(f"EN: {sentence}\n")

        try:
            # Step 1: Translate
            print("Step 1: Translation...")
            start_time = asyncio.get_event_loop().time()
            translation = await translator.translate(sentence, context)
            translation_time = asyncio.get_event_loop().time() - start_time
            total_translation_time += translation_time

            print(f"RU: {translation}")
            print(f"Translation latency: {translation_time:.2f}s\n")

            # Update context
            context.append(sentence)
            if len(context) > 10:
                context.pop(0)

            # Step 2: Generate speech
            print("Step 2: TTS generation...")
            start_time = asyncio.get_event_loop().time()
            audio_bytes = await tts.generate_speech(translation)
            tts_time = asyncio.get_event_loop().time() - start_time
            total_tts_time += tts_time

            print(f"Audio size: {len(audio_bytes)} bytes")
            print(f"TTS latency: {tts_time:.2f}s\n")

            # Step 3: Save audio
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"pipeline_{i:02d}_{timestamp}.wav"
            with open(output_file, "wb") as f:
                f.write(audio_bytes)

            print(f"✓ Saved: {output_file}")

            # Pipeline stats
            total_time = translation_time + tts_time
            print(f"\nTotal pipeline time: {total_time:.2f}s")
            print(f"  - Translation: {translation_time:.2f}s ({translation_time/total_time*100:.1f}%)")
            print(f"  - TTS: {tts_time:.2f}s ({tts_time/total_time*100:.1f}%)")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Final statistics
    print("\n" + "=" * 80)
    print("PIPELINE TEST COMPLETED")
    print("=" * 80)
    print(f"\nProcessed: {len(sentences)} sentences")
    print(f"Output directory: {output_dir}\n")

    print("Average latencies:")
    print(f"  - Translation: {total_translation_time/len(sentences):.2f}s")
    print(f"  - TTS: {total_tts_time/len(sentences):.2f}s")
    print(f"  - Total pipeline: {(total_translation_time + total_tts_time)/len(sentences):.2f}s")

    print("\nNext steps:")
    print("  1. Review translations for accuracy and context preservation")
    print("  2. Listen to audio files for voice quality and pronunciation")
    print("  3. Adjust parameters in config.yaml if needed:")
    print("     - llm.model (for different translation models)")
    print("     - llm.temperature (for translation creativity)")
    print("     - tts.speed (for speech speed)")
    print("     - tts.temperature (for voice variation)")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
