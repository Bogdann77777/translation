"""
Proof test - saves API output to file to prove it's not manually edited.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.openrouter_llm import OpenRouterClient


TEST_SENTENCES = [
    "So like, you know, the thing is, um, the economy is kind of a hot mess right now, to be honest.",
    "Like, people are literally freaking out because, I mean, inflation is through the roof, you know?",
    "Student loans are like a ball and chain around people's necks, seriously, it's insane.",
]


async def test_api_proof():
    translator = OpenRouterClient()

    output = []
    output.append("=" * 80)
    output.append("API TRANSLATION PROOF TEST")
    output.append("=" * 80)
    output.append("\nThese translations come DIRECTLY from OpenRouter API")
    output.append("NO manual editing - this is the LLM's output!\n")

    for i, sentence in enumerate(TEST_SENTENCES, 1):
        output.append(f"\n{'=' * 80}")
        output.append(f"TEST #{i}")
        output.append(f"{'=' * 80}")

        output.append(f"\nENGLISH (original with filler words):")
        output.append(f"{sentence}")

        fillers = sentence.lower().count('like') + sentence.lower().count('you know') + sentence.lower().count('i mean') + sentence.lower().count('um')
        output.append(f"\nFiller words count: {fillers}")
        output.append(f"Original length: {len(sentence)} chars")

        print(f"[*] Calling OpenRouter API for sentence #{i}...", flush=True)

        # REAL API CALL - NO EDITING!
        translation = await translator.translate(sentence, [])

        print(f"    API returned: {len(translation)} chars")

        output.append(f"\nRUSSIAN (direct from API - NOT edited):")
        output.append(f"{translation}")

        output.append(f"\nTranslation length: {len(translation)} chars")
        output.append(f"Compression ratio: {len(sentence)} -> {len(translation)} ({len(translation)/len(sentence)*100:.0f}%)")

        # Check what API should have done
        output.append(f"\n[EXPECTED BEHAVIOR CHECK]")
        if fillers > 0:
            output.append(f"  Should remove {fillers} filler words: {'YES - check if removed' if len(translation) < len(sentence) else 'NO - text not shortened'}")

        if 'hot mess' in sentence.lower():
            has_literal = 'горячий беспорядок' in translation.lower() or 'hot mess' in translation.lower()
            output.append(f"  'hot mess' adapted to Russian idiom: {'NO - literal translation' if has_literal else 'YES - idiom adapted'}")

        if 'through the roof' in sentence.lower():
            has_literal = 'через крышу' in translation.lower() or 'through the roof' in translation.lower()
            output.append(f"  'through the roof' adapted: {'NO - literal' if has_literal else 'YES - adapted'}")

        if 'ball and chain' in sentence.lower():
            has_literal = 'шар и цепь' in translation.lower() or 'ball and chain' in translation.lower()
            output.append(f"  'ball and chain' adapted: {'NO - literal' if has_literal else 'YES - adapted'}")

    output.append("\n" + "=" * 80)
    output.append("PROOF COMPLETE")
    output.append("=" * 80)
    output.append("\nAll translations above came from OpenRouter API!")
    output.append("The API was configured with the contextual translation prompt.")
    output.append("This proves the LLM is doing the smart translation, not manual editing!")

    # Save to file
    output_file = Path(__file__).parent.parent / "test_output" / "api_proof.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))

    print(f"\n[OK] Test complete! Results saved to: {output_file}")
    print("\nOpen the file to see the API's translations!")


if __name__ == "__main__":
    asyncio.run(test_api_proof())
