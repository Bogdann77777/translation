"""
Test script for calibrating Translation component without audio streaming.
Tests the OpenRouter LLM translation quality and parameters.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.openrouter_llm import OpenRouterLLM
from app.config import load_config


async def test_translation():
    """Test translation with sample text."""
    print("=" * 80)
    print("TRANSLATION CALIBRATION TEST")
    print("=" * 80)

    # Load config
    config = load_config()

    # Initialize translator
    translator = OpenRouterLLM(config)

    # Load sample text
    sample_file = Path(__file__).parent.parent / "test_data" / "sample_text.txt"
    with open(sample_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into sentences
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]

    print(f"\nLoaded {len(sentences)} sentences from sample text\n")

    # Test translation sentence by sentence
    context = []
    for i, sentence in enumerate(sentences, 1):
        print(f"\n{'─' * 80}")
        print(f"Sentence {i}/{len(sentences)}")
        print(f"{'─' * 80}")
        print(f"EN: {sentence}")

        try:
            # Translate with context
            translation = await translator.translate(sentence, context)

            print(f"RU: {translation}")

            # Update context (keep last 10)
            context.append(sentence)
            if len(context) > 10:
                context.pop(0)

        except Exception as e:
            print(f"ERROR: {e}")

    print("\n" + "=" * 80)
    print("TRANSLATION TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_translation())
