"""
Test contextual translation of colloquial American speech with slang and filler words.
Tests the translator's ability to convey MEANING, not literal words.
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


async def test_colloquial_translation():
    """Test translation of colloquial American speech."""

    print("=" * 80)
    print("COLLOQUIAL AMERICAN SPEECH TRANSLATION TEST")
    print("Testing contextual translation with slang, filler words, and idioms")
    print("=" * 80)

    # Load config
    try:
        config = load_config()
    except Exception as e:
        print(f"\n[ERROR] Failed to load config: {e}")
        return

    # Initialize translator
    print("\n[*] Initializing contextual translator...")
    try:
        translator = OpenRouterClient()
        print("[OK] Translator ready")
        print("[*] Mode: Contextual (removes filler words, translates meaning)")
    except Exception as e:
        print(f"[ERROR] Failed to initialize translator: {e}")
        return

    # Load colloquial text
    text_file = Path(__file__).parent.parent / "test_data" / "colloquial_american_text.txt"
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into sentences
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    print(f"[*] Loaded {len(sentences)} sentences with American slang\n")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "test_output" / "colloquial_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Output directory: {output_dir}\n")

    # Stats
    context = []
    results = []
    translation_times = []
    tts_times = []

    test_start = time.time()

    # Process each sentence
    for i, sentence in enumerate(sentences, 1):
        print(f"\n{'=' * 80}")
        print(f"SENTENCE #{i}/{len(sentences)}")
        print(f"{'=' * 80}")
        print(f"\n[EN ORIGINAL] {sentence}")

        # Count filler words
        filler_words = ['like', 'you know', 'I mean', 'um', 'basically', 'kind of', 'sort of', 'literally', 'actually']
        filler_count = sum(sentence.lower().count(filler) for filler in filler_words)
        if filler_count > 0:
            print(f"[ANALYSIS] Contains {filler_count} filler words/phrases")

        try:
            # Step 1: Translation
            print("\n[*] Translating (contextual mode)...", end=" ", flush=True)
            trans_start = time.time()
            translation = await translator.translate(sentence, context)
            trans_time = time.time() - trans_start
            translation_times.append(trans_time)

            print(f"Done ({trans_time:.2f}s)")
            print(f"[RU CLEANED] <{len(translation)} chars>")

            # Update context
            context.append(sentence)
            if len(context) > 10:
                context.pop(0)

            # Step 2: TTS Generation
            print("[*] Generating audio...", end=" ", flush=True)
            tts_start = time.time()

            # Generate audio
            tts = gTTS(text=translation, lang='ru', slow=False)
            audio_file = output_dir / f"audio_{i:02d}.mp3"
            tts.save(str(audio_file))

            tts_time = time.time() - tts_start
            tts_times.append(tts_time)

            print(f"Done ({tts_time:.2f}s)")
            print(f"[SAVED] {audio_file.name}")

            # Save results
            results.append({
                'num': i,
                'en_original': sentence,
                'ru_translation': translation,
                'filler_count': filler_count,
                'audio_file': audio_file.name,
                'trans_time': trans_time,
                'tts_time': tts_time,
                'total_time': trans_time + tts_time
            })

            print(f"[OK] Total time: {trans_time + tts_time:.2f}s")

        except Exception as e:
            print(f"\n[ERROR] Sentence #{i}: {e}")
            import traceback
            traceback.print_exc()

    test_duration = time.time() - test_start

    # Save detailed results
    results_file = output_dir / "translation_analysis.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("COLLOQUIAL AMERICAN SPEECH - CONTEXTUAL TRANSLATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write("\nTRANSLATION MODE: Contextual (meaning-based, filler removal)\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            f.write(f"SENTENCE #{result['num']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"ENGLISH (ORIGINAL):\n{result['en_original']}\n\n")
            f.write(f"RUSSIAN (CLEANED):\n{result['ru_translation']}\n\n")
            f.write(f"Filler words in original: {result['filler_count']}\n")
            f.write(f"Audio file: {result['audio_file']}\n")
            f.write(f"Translation time: {result['trans_time']:.2f}s\n")
            f.write(f"TTS time: {result['tts_time']:.2f}s\n")
            f.write(f"Total time: {result['total_time']:.2f}s\n")
            f.write("\n" + "=" * 80 + "\n\n")

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total_fillers = sum(r['filler_count'] for r in results)
    print(f"\n[ANALYSIS] Filler Words:")
    print(f"  Total filler words/phrases detected: {total_fillers}")
    print(f"  Average per sentence: {total_fillers / len(results):.1f}")
    print(f"  Translation mode: CONTEXTUAL (fillers removed)")

    print(f"\n[STATS] Overall:")
    print(f"  Sentences processed: {len(results)}")
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

    print(f"\n[OUTPUT] Files:")
    print(f"  Directory: {output_dir}")
    print(f"  Audio files: {len(results)} x MP3")
    print(f"  Analysis: {results_file.name}")

    print("\n" + "=" * 80)
    print("WHAT TO CHECK")
    print("=" * 80)
    print("\n1. TRANSLATION QUALITY:")
    print("   - Open translation_analysis.txt")
    print("   - Compare ENGLISH (ORIGINAL) vs RUSSIAN (CLEANED)")
    print("   - Check if filler words are removed")
    print("   - Check if slang/idioms are adapted to Russian")
    print("   - Check if meaning is preserved (not literal)")
    print("\n2. AUDIO QUALITY:")
    print(f"   - Open folder: {output_dir}")
    print("   - Listen to audio_01.mp3 through audio_15.mp3")
    print("   - Check pronunciation of Russian text")
    print("   - Check clarity and naturalness")
    print("   - Check for artifacts or unwanted sounds")
    print("\n3. EXAMPLES TO REVIEW:")
    print("   - 'hot mess' -> should be adapted Russian idiom")
    print("   - 'through the roof' -> should be Russian equivalent")
    print("   - 'ball and chain' -> should be Russian idiom")
    print("   - Multiple 'like, you know' -> should be removed")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_colloquial_translation())
