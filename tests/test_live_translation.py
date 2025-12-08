"""
Live translation test - shows real-time translation to verify it's the API doing the work.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.openrouter_llm import OpenRouterClient


# Very challenging sentences with slang and filler words
TEST_SENTENCES = [
    "So like, you know, the thing is, um, the economy is kind of a hot mess right now, to be honest.",
    "Like, people are literally freaking out because, I mean, inflation is through the roof, you know?",
    "Student loans are like a ball and chain around people's necks, seriously, it's insane.",
]


async def test_live():
    print("=" * 80)
    print("LIVE TRANSLATION TEST")
    print("Watch the API translate in real-time!")
    print("=" * 80)

    translator = OpenRouterClient()
    print("\n[*] Translator initialized (using OpenRouter API)")
    print("[*] Model: Claude/Mistral via API call")
    print("\nNOTE: The translations below come DIRECTLY from the API, not edited by me!\n")

    for i, sentence in enumerate(TEST_SENTENCES, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST #{i}")
        print(f"{'=' * 80}")

        print(f"\nENGLISH (with filler words):")
        print(f"  {sentence}")

        # Count filler words
        fillers = sentence.lower().count('like') + sentence.lower().count('you know') + sentence.lower().count('i mean') + sentence.lower().count('um')
        print(f"\nFiller words detected: {fillers}")
        print(f"Original length: {len(sentence)} chars")

        print(f"\n[*] Calling OpenRouter API...", end=" ", flush=True)

        # THIS IS THE REAL API CALL - NO EDITING!
        translation = await translator.translate(sentence, [])

        print("Done!")

        print(f"\nRUSSIAN (from API):")
        print(f"  {translation}")
        print(f"\nTranslated length: {len(translation)} chars")
        print(f"Compression: {len(sentence)} -> {len(translation)} ({len(translation)/len(sentence)*100:.0f}%)")

        # Show what should be removed/adapted
        print(f"\n[ANALYSIS]")
        if 'like' in sentence.lower() or 'you know' in sentence.lower() or 'i mean' in sentence.lower():
            print(f"  Filler words should be removed: YES")
        if 'hot mess' in sentence.lower():
            print(f"  'hot mess' should become Russian idiom")
        if 'through the roof' in sentence.lower():
            print(f"  'through the roof' should become Russian idiom")
        if 'ball and chain' in sentence.lower():
            print(f"  'ball and chain' should become Russian idiom")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nAll translations above came DIRECTLY from OpenRouter API!")
    print("No manual editing was done - this is the LLM's work!")


if __name__ == "__main__":
    asyncio.run(test_live())
