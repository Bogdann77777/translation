# CosyVoice3 Installation Guide

This guide walks you through installing CosyVoice3 for zero-shot TTS in the translation project.

## Prerequisites

- Python 3.10
- CUDA 12.8 (for GPU support)
- Existing translator environment activated

## Installation Steps

### 1. Install CosyVoice Dependencies

```bash
# Install required packages
pip install modelscope torchaudio

# Install sox (audio processing library)
# Windows: Download from https://sourceforge.net/projects/sox/files/sox/
# Linux: sudo apt-get install sox libsox-dev
```

### 2. Clone CosyVoice Repository (Third-party dependency)

```bash
# Navigate to project root
cd E:/crewai/translator

# Clone CosyVoice repo into third_party folder
mkdir -p third_party
cd third_party
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# Install CosyVoice requirements (in existing environment)
pip install -r requirements.txt
```

### 3. Download Pretrained Model

```python
# Run this Python script to download model
from modelscope import snapshot_download

snapshot_download(
    'FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir='E:/crewai/translator/models/CosyVoice3-0.5B'
)
```

Or download manually from: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512

### 4. Update Config

Model will be loaded from: `E:/crewai/translator/models/CosyVoice3-0.5B`

Voice samples location: `E:/crewai/translator/voice_samples/`

## Verify Installation

```python
import sys
sys.path.append('third_party/CosyVoice/third_party/Matcha-TTS')
from third_party.CosyVoice.cosyvoice.cli.cosyvoice import AutoModel

# Load model
cosyvoice = AutoModel(model_dir='models/CosyVoice3-0.5B')
print("CosyVoice3 loaded successfully!")
```

## Troubleshooting

**Import Error**: Make sure to add Matcha-TTS to Python path before importing
**Model Not Found**: Check model_dir path is correct
**CUDA OOM**: Use GPU 1 (TTS on GPU 1, Whisper on GPU 0)

## Resources

- GitHub: https://github.com/FunAudioLLM/CosyVoice
- Hugging Face: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- Documentation: https://cosyvoice.org/
