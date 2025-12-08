"""
Test Translation + TTS pipeline with audio generation.
Uses Google TTS for quick testing.
"""

import asyncio
import sys
from pathlib import Path
import time
from gtts import gTTS

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.openrouter_llm import OpenRouterClient
from app.config import load_config


# Test sentences (shorter list for quick audio test)
TEST_SENTENCES = [
    "Artificial intelligence is transforming the way we live and work in the modern world.",
    "Machine learning algorithms can now recognize patterns in data that humans might miss.",
    "Natural language processing has made it possible for computers to understand human language.",
    "Deep learning models require vast amounts of data and computational power to train effectively.",
    "Neural networks are inspired by the structure and function of the human brain.",
    "Quantum computing may revolutionize AI by solving problems that are currently intractable.",
    "The future of AI promises both tremendous opportunities and significant challenges for society.",
]


async def test_tts_audio():
    """Test full Translation + TTS pipeline with audio generation."""

    print("=" * 80)
    print("TRANSLATION + TTS AUDIO TEST")
    print("Testing: English Text -> Translation -> Russian Audio")
    print("=" * 80)

    # Load config
    try:
        config = load_config()
    except Exception as e:
        print(f"\n[ERROR] Failed to load config: {e}")
        return

    # Initialize translator
    print("\n[*] Initializing translator...")
    try:
        translator = OpenRouterClient()
        print("[OK] Translator ready")
    except Exception as e:
        print(f"[ERROR] Failed to initialize translator: {e}")
        return

    # Create output directory
    output_dir = Path(__file__).parent.parent / "test_output" / "tts_audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Output directory: {output_dir}\n")

    # Stats
    context = []
    translation_times = []
    tts_times = []
    results = []

    test_start = time.time()

    # Process each sentence
    for i, sentence in enumerate(TEST_SENTENCES, 1):
        print(f"\n{'=' * 80}")
        print(f"BATCH #{i}/{len(TEST_SENTENCES)}")
        print(f"{'=' * 80}")
        print(f"\n[EN] {sentence}")

        try:
            # Step 1: Translation
            print("[*] Translating...", end=" ", flush=True)
            trans_start = time.time()
            translation = await translator.translate(sentence, context)
            trans_time = time.time() - trans_start
            translation_times.append(trans_time)

            print(f"Done ({trans_time:.2f}s)")
            print(f"[RU] <{len(translation)} chars>")

            # Update context
            context.append(sentence)
            if len(context) > 10:
                context.pop(0)

            # Step 2: TTS Generation
            print("[*] Generating audio...", end=" ", flush=True)
            tts_start = time.time()

            # Generate audio using Google TTS
            tts = gTTS(text=translation, lang='ru', slow=False)

            # Save to file
            audio_file = output_dir / f"audio_{i:02d}.mp3"
            tts.save(str(audio_file))

            tts_time = time.time() - tts_start
            tts_times.append(tts_time)

            print(f"Done ({tts_time:.2f}s)")
            print(f"[SAVED] {audio_file.name}")

            # Save results
            results.append({
                'batch': i,
                'en': sentence,
                'ru': translation,
                'audio_file': audio_file.name,
                'trans_time': trans_time,
                'tts_time': tts_time,
                'total_time': trans_time + tts_time
            })

            print(f"[OK] Total time: {trans_time + tts_time:.2f}s")

        except Exception as e:
            print(f"\n[ERROR] Batch #{i}: {e}")
            import traceback
            traceback.print_exc()

    test_duration = time.time() - test_start

    # Save text results
    results_file = output_dir / "results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TRANSLATION + TTS RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            f.write(f"BATCH #{result['batch']}\n")
            f.write(f"EN: {result['en']}\n")
            f.write(f"RU: {result['ru']}\n")
            f.write(f"Audio: {result['audio_file']}\n")
            f.write(f"Translation: {result['trans_time']:.2f}s\n")
            f.write(f"TTS: {result['tts_time']:.2f}s\n")
            f.write(f"Total: {result['total_time']:.2f}s\n")
            f.write("\n" + "-" * 80 + "\n\n")

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    print(f"\n[STATS] Overall:")
    print(f"  Batches processed: {len(results)}")
    print(f"  Total test time: {test_duration:.2f}s")

    if translation_times:
        avg_trans = sum(translation_times) / len(translation_times)
        print(f"\n[TRANSLATION] Times:")
        print(f"  Average: {avg_trans:.2f}s")
        print(f"  Min: {min(translation_times):.2f}s")
        print(f"  Max: {max(translation_times):.2f}s")

    if tts_times:
        avg_tts = sum(tts_times) / len(tts_times)
        print(f"\n[TTS] Times:")
        print(f"  Average: {avg_tts:.2f}s")
        print(f"  Min: {min(tts_times):.2f}s")
        print(f"  Max: {max(tts_times):.2f}s")

    total_times = [r['total_time'] for r in results]
    if total_times:
        avg_total = sum(total_times) / len(total_times)
        print(f"\n[PIPELINE] Total Times:")
        print(f"  Average: {avg_total:.2f}s")
        print(f"  Min: {min(total_times):.2f}s")
        print(f"  Max: {max(total_times):.2f}s")

    print(f"\n[OUTPUT] Files saved to:")
    print(f"  Directory: {output_dir}")
    print(f"  Audio files: {len(results)} x MP3")
    print(f"  Results: {results_file.name}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Open the output directory:")
    print(f"   {output_dir}")
    print("\n2. Listen to the audio files (audio_01.mp3, audio_02.mp3, etc.)")
    print("\n3. Check for:")
    print("   - Voice quality and naturalness")
    print("   - Pronunciation accuracy")
    print("   - Unwanted sounds or artifacts")
    print("   - Speech clarity and understandability")
    print("\n4. Compare with the Russian text in results.txt")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_tts_audio())
