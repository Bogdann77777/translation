# Installation Instructions

## System Requirements

- Python 3.11
- NVIDIA GPU with CUDA 12.8 support
- RTX 5060 Ti or newer (Compute Capability sm_120)

## Installation Steps

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install PyTorch Nightly (Required for RTX 5060 Ti)

```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchaudio torchvision
```

**Important:** Regular PyTorch releases do not support RTX 5060 Ti (sm_120). You must use nightly builds.

### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 4. Fix TTS Compatibility with PyTorch 2.10+

After installation, apply this patch to `venv\Lib\site-packages\TTS\utils\io.py`:

Find lines 51 and 54, and add `weights_only=False` parameter:

**Before:**
```python
return torch.load(f, map_location=map_location, **kwargs)
```

**After:**
```python
return torch.load(f, map_location=map_location, weights_only=False, **kwargs)
```

This is required because PyTorch 2.10+ changed the default value of `weights_only` from `False` to `True`.

### 5. Configure Environment

1. Copy `.env.example` to `.env` (if exists)
2. Add your API keys:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```

3. Update `config.yaml` with your settings

### 6. Verify Installation

```bash
venv\Scripts\python.exe -c "from TTS.api import TTS; import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); torch.cuda.set_device(1); tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=True); print('SUCCESS')"
```

You should see:
```
PyTorch: 2.10.0.dev20251108+cu128
CUDA: True
TTS LOADED SUCCESS
```

### 7. Run Application

```bash
START.bat
```

## Troubleshooting

### Issue: `ImportError: cannot import name 'TrainerConfig' from 'trainer'`

**Solution:** Reinstall TTS:
```bash
pip uninstall -y trainer TTS
pip install TTS==0.22.0
```

### Issue: `CUDA error: no kernel image is available`

**Solution:** You need PyTorch nightly build with CUDA 12.8 support.
```bash
pip uninstall -y torch torchaudio torchvision
pip install --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchaudio torchvision
```

### Issue: `Weights only load failed`

**Solution:** Apply the TTS patch mentioned in Step 4.

## Notes

- TTS 0.22.0 is required (newer versions have Unicode encoding bugs on Windows)
- PyTorch nightly is required for RTX 5060 Ti support
- GPU 0 is used for Whisper, GPU 1 is used for XTTS (configurable in `config.yaml`)
