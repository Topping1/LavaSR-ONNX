# LavaSR-ONNX Runtime

A small, PyTorch-free runtime for speech enhancement based on [LavaSR](https://github.com/ysharma3501/LavaSR).

This project keeps the original LavaSR idea and model structure, but runs inference with **ONNX Runtime + NumPy/SciPy + SoundFile** instead of PyTorch.

## Original LavaSR

This app is derived from the original [ysharma3501/LavaSR](https://github.com/ysharma3501/LavaSR) repository.

From the original repository, LavaSR is a lightweight speech enhancement / restoration system for improving low-quality speech. The upstream project describes support for speech bandwidth extension up to 48 kHz, optional denoising, and use cases such as TTS enhancement, real-time enhancement, and dataset restoration. The repository also explains that the method uses a Vocos-based architecture for bandwidth extension together with a Linkwitz-Riley-inspired refiner, and acknowledges UL-UNAS for the denoiser component.

This runtime-only version focuses on **inference only**. Model export/conversion is out of scope here.

## What this repository contains

This app runs the pipeline with:

- **ONNX Runtime** for the denoiser core and enhancer neural networks
- **NumPy / SciPy** for STFT, ISTFT, resampling, mel features, and spectral-domain glue logic
- **SoundFile** for WAV/FLAC audio I/O

The current inference flow is:

1. Load audio at 16 kHz
2. Optionally denoise with the ONNX ULUNAS-derived denoiser core
3. Resample to 48 kHz
4. Run the ONNX enhancer backbone and ONNX spectrogram head
5. Reconstruct waveform and apply the final spectral merge

## Files needed to run `main.py`

Expected runtime file structure:

```text
.
├── main.py
├── lavasr_core.py
├── config.yaml
├── denoiser_core_legacy_fixed63.onnx
├── enhancer_backbone.onnx
├── enhancer_backbone.onnx.data
├── enhancer_spec_head.onnx
└── enhancer_spec_head.onnx.data
```

## Model files

The ONNX files are hosted in the release assets here:

- Release page: <https://github.com/Topping1/LavaSR-ONNX/releases/tag/Alpha-v0.1>
- `denoiser_core_legacy_fixed63.onnx`: <https://github.com/Topping1/LavaSR-ONNX/releases/download/Alpha-v0.1/denoiser_core_legacy_fixed63.onnx>
- `enhancer_backbone.onnx`: <https://github.com/Topping1/LavaSR-ONNX/releases/download/Alpha-v0.1/enhancer_backbone.onnx>
- `enhancer_backbone.onnx.data`: <https://github.com/Topping1/LavaSR-ONNX/releases/download/Alpha-v0.1/enhancer_backbone.onnx.data>
- `enhancer_spec_head.onnx`: <https://github.com/Topping1/LavaSR-ONNX/releases/download/Alpha-v0.1/enhancer_spec_head.onnx>
- `enhancer_spec_head.onnx.data`: <https://github.com/Topping1/LavaSR-ONNX/releases/download/Alpha-v0.1/enhancer_spec_head.onnx.data>

## Python dependencies

Install the runtime dependencies with:

```bash
pip install numpy scipy soundfile onnxruntime
```

For GPU inference, use the ONNX Runtime package appropriate for your platform instead of the default CPU package.

## Usage

Basic run:

```bash
python main.py input.wav -o output.wav
```

Run with denoising enabled:

```bash
python main.py input.wav -o output.wav --denoise
```

## Notes

- This repository is intended for **runtime inference**.
- It does **not** require PyTorch for normal execution.
- The denoiser ONNX model is used with fixed-size internal chunks in the runtime code.
- The original LavaSR project remains the reference implementation for the model and overall method.

## Acknowledgment

- Original project: [ysharma3501/LavaSR](https://github.com/ysharma3501/LavaSR)
