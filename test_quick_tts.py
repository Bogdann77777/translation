"""Quick TTS test"""
import sys
import io

# Force UTF-8 encoding for Windows (fix TTS library charmap error)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import asyncio
sys.path.insert(0, ".")

from app.components.xtts_engine import XTTSEngine

async def test():
    print("Initializing XTTS...")
    tts = XTTSEngine()

    print("Synthesizing...")
    audio = await tts.synthesize("Привет мир!")

    print(f"SUCCESS! Got {len(audio)} bytes of audio")

    with open("test_output/quick_test.wav", "wb") as f:
        f.write(audio)
    print("Saved to test_output/quick_test.wav")

if __name__ == "__main__":
    asyncio.run(test())
