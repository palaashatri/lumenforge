# LumenForge

Desktop Java Swing application for Java-native ONNX Runtime workflows with automatic GPU/CPU provider fallback.

- Text → Image: Stable Diffusion v1.5 UNet + Real-ESRGAN with fully automatic downloads

<img width="2552" height="2026" alt="image" src="https://github.com/user-attachments/assets/8b7a8783-c795-4a2f-844c-beccc9d6855d" />


## Features

- Native look-and-feel handling:
  - macOS: system-native look-and-feel
  - Windows/Linux: FlatLaf with automatic system dark/light detection
- High-performance async execution using virtual threads
- Model downloader module with progress reporting
- Auto-download when a selected model is missing in task tabs
- Model Manager downloads only on explicit **Download / Redownload** button click
- Model Manager includes additional ONNX pipeline assets (UNet/Text Encoder/VAE/Safety Checker) for one-click download
- Manual ONNX model import per selected row in Model Manager
- Prompt library: saved presets + tags + search
- Negative prompts + prompt weighting
- Seed control + reproducibility presets
- Batch generation settings (N images per prompt)
- Aspect ratio presets + custom size fields
- Upscale toggle (Real-ESRGAN) with before/after preview (UI wiring)
- Style presets (cinematic, sketch, product, etc.)
- History gallery with metadata (seed, model, settings)
- Export pipeline logs for debugging
- Local model storage in `~/.lumenforge-models`
- Execution provider fallback by OS:
  - macOS: CoreML → CPU
  - Windows: DirectML → CUDA → CPU
  - Linux: CUDA → ROCm → CPU

## Build

```bash
mvn -DskipTests compile
```

Enable GPU runtime artifact (Windows/Linux):

```bash
mvn -Donnx.gpu=true -DskipTests compile
```

## Run

```bash
mvn exec:java
```

## Test

```bash
mvn clean test
```

## Notes

- Runtime is Java-only ONNX Runtime execution with GPU fallback (no Python bridge).
- Override provider order via JVM property: `-Dlumenforge.ep=cpu|coreml|directml|cuda|rocm`.
- Task tabs show preview images when an output artifact is generated, with **Open Output** to launch the file.
- SD Turbo UNet remains experimental and may fail on CPU-only environments.
- If a model requires external tensor files (e.g. `weights.pb`), import the complete ONNX bundle into the model directory.
