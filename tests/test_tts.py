"""
Test script for calibrating TTS (Text-to-Speech) component.
Tests XTTS-v2 voice quality, speed, and audio output.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.xtts_engine import XTTSEngine
from app.config import load_config


# Sample Russian sentences for TTS testing
RUSSIAN_TEST_SENTENCES = [
    "Искусственный интеллект меняет наш мир.",
    "Машинное обучение находит скрытые закономерности в данных.",
    "Обработка естественного языка позволяет компьютерам понимать человеческую речь.",
    "Компьютерное зрение распознаёт объекты на изображениях с высокой точностью.",
    "Глубокое обучение требует больших объёмов данных и вычислительных ресурсов.",
]


async def test_tts():
    """Test TTS with Russian sentences."""
    print("=" * 80)
    print("TTS CALIBRATION TEST (XTTS-v2)")
    print("=" * 80)

    # Load config
    config = load_config()

    # Initialize TTS engine
    print("\nInitializing XTTS engine...")
    tts = XTTSEngine(config)

    print(f"Voice sample: {config['tts']['voice_sample_path']}")
    print(f"Language: {config['tts']['language']}")
    print(f"Speed: {config['tts'].get('speed', 1.0)}")
    print(f"Temperature: {config['tts'].get('temperature', 0.7)}")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "test_output"
    output_dir.mkdir(exist_ok=True)

    print(f"\nOutput directory: {output_dir}\n")

    # Test each sentence
    for i, sentence in enumerate(RUSSIAN_TEST_SENTENCES, 1):
        print(f"\n{'─' * 80}")
        print(f"Test {i}/{len(RUSSIAN_TEST_SENTENCES)}")
        print(f"{'─' * 80}")
        print(f"Text: {sentence}")

        try:
            # Generate speech
            start_time = asyncio.get_event_loop().time()
            audio_bytes = await tts.generate_speech(sentence)
            end_time = asyncio.get_event_loop().time()

            latency = end_time - start_time

            print(f"Audio size: {len(audio_bytes)} bytes")
            print(f"Latency: {latency:.2f}s")

            # Save audio file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"tts_test_{i}_{timestamp}.wav"
            with open(output_file, "wb") as f:
                f.write(audio_bytes)

            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"ERROR: {e}")

    print("\n" + "=" * 80)
    print("TTS TEST COMPLETED")
    print(f"Audio files saved to: {output_dir}")
    print("=" * 80)
    print("\nListen to the generated audio files to evaluate:")
    print("  - Voice quality and naturalness")
    print("  - Pronunciation accuracy")
    print("  - Speech speed")
    print("  - Intonation and emotion")


if __name__ == "__main__":
    asyncio.run(test_tts())
