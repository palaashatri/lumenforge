# JForge

[![Build JForge](https://github.com/palaashatri/jforge/actions/workflows/build.yml/badge.svg)](https://github.com/palaashatri/jforge/actions/workflows/build.yml)

Desktop Java Swing application for ONNX Runtime inference with intelligent GPU acceleration across NVIDIA, Apple, Intel, and AMD hardware.


## Features

### Image Generation
- **Text → Image**: Stable Diffusion v1.5, SD Turbo, SDXL Turbo, SDXL Base 1.0, SD 3.5 (MMDiT)
- **Image → Image**: Img2Img with adjustable strength + inpainting
- **Image Upscaling**: Real-ESRGAN 4× super-resolution with before/after preview
- **Prompt library**: saved presets, tags, and search
- Negative prompts + prompt weighting
- Seed control + reproducibility
- Batch generation (N images per prompt)
- Aspect ratio presets + custom size fields
- Style presets (cinematic, sketch, product, etc.)
- History gallery with metadata (seed, model, settings)
- Export pipeline logs for debugging

### SD 3.5 Support
- MMDiT transformer architecture with Flow Matching Euler scheduler
- Triple text encoding: CLIP-L (768d) + CLIP-G (1280d) + T5-XXL (4096d)
- Built-in T5 tokenizer (SentencePiece/Unigram with Viterbi segmentation)
- 16-channel latent space

### Model Management
- **Automatic HuggingFace discovery**: finds ONNX and PyTorch Stable Diffusion + ESRGAN models
- **One-click download** with resume, retry, and stall detection
- **PyTorch → ONNX auto-conversion**: downloading a PyTorch model triggers automatic conversion via managed Python venv
- Manual ONNX model import
- Gated model support with HuggingFace token authentication
- Model metadata ingestion with version and tags (`model.json` when present)
- Pre-run model compatibility checks for scheduler/sampler constraints
- Local model storage in `~/.jforge-models`

### GPU Acceleration
Intelligent execution provider selection — JForge probes available EPs at runtime and picks the best one:

| Platform | Priority (highest → lowest) |
|---|---|
| **macOS** | CoreML (GPU + ANE + CPU) → CPU |
| **Windows** | TensorRT-RTX → TensorRT → CUDA → DirectML → OpenVINO → CPU |
| **Linux** | TensorRT → CUDA → ROCm → OpenVINO → CPU |

Override with `-Djforge.ep=cuda` (or any EP key) to force a specific provider.

### UI
- Top task tabs for Text -> Image, Image -> Image, Inpaint, Upscale, Settings, and Models
- Bottom thumbnail history strip with click-to-inspect metadata and right-click open folder
- Per-output metadata JSON sidecars written alongside generated images
- Native look-and-feel: system-native on macOS, FlatLaf with dark/light detection on Windows/Linux
- High-performance async execution using virtual threads
- Per-step progress with timing and ETA
- Session and tokenizer caching for fast repeated inference

### Architecture (In Progress)
- `core`: scheduler, sampler, per-step callback, tensor abstractions
- `tasks`: unified `ForgeTask` API and task implementations
- `models`: `ForgeModel`, model typing, metadata and compatibility checks
- `runtime`: `SessionManager` for ONNX Runtime session caching and device information
- `ui`: tabbed workflow shell plus generation history strip

## Downloads

Pre-built fat JARs are available from [GitHub Releases](https://github.com/palaashatri/jforge/releases):

| JAR | GPU Support | Use When |
|---|---|---|
| `jforge-universal.jar` | macOS CoreML (M-series GPU/ANE), CPU everywhere | macOS, or Windows/Linux without NVIDIA GPU |
| `jforge-nvidia.jar` | CUDA + TensorRT (Windows/Linux) | Windows/Linux with NVIDIA GPU + CUDA installed |

> **Note**: DirectML (AMD/Intel on Windows), OpenVINO (Intel), and ROCm (AMD on Linux) are auto-detected at runtime if the native libraries are installed on the system. The universal JAR handles this automatically.

```bash
# Run any variant
java -jar jforge-universal.jar
java -jar jforge-nvidia.jar
```

## Build from Source

Requires **Java 21+** and **Maven 3.8+**.

```bash
# Universal build (CPU + CoreML)
mvn clean package -DskipTests

# NVIDIA GPU build (CUDA + TensorRT)
mvn clean package -DskipTests -Dort.artifactId=onnxruntime_gpu

# Force CPU-only
mvn clean package -DskipTests -Dort.artifactId=onnxruntime
```

### Run from source

```bash
mvn clean compile exec:java
```

### Run tests

```bash
mvn clean test
```

## Packaging

Cross-platform packaging scripts are available in `scripts/`:

```bash
# macOS .app bundle
./scripts/package-macos-app.sh 1.0.0

# Linux app-image (and optional AppImage if appimagetool is installed)
./scripts/package-linux-appimage.sh 1.0.0
```

```powershell
# Windows .exe (PowerShell)
./scripts/package-windows-exe.ps1 -Version 1.0.0
```

All scripts use `jpackage` and expect the shaded jar to exist in `target/`.

## CI / CD

GitHub Actions builds both JAR variants on every push to `main` and PR. Pushing a version tag (e.g. `v1.0.0`) creates a GitHub Release with both JARs attached.

See [.github/workflows/build.yml](.github/workflows/build.yml) for details.

## Requirements

- **Java 21** or later
- **macOS 10.15+** for CoreML acceleration (M-series recommended)
- **CUDA 12 + cuDNN** for NVIDIA GPU acceleration (RTX 30xx+ recommended)
- **Python 3.8+** (optional) for PyTorch → ONNX model conversion

## Notes

- Runtime is pure Java — ONNX Runtime execution with GPU fallback. No Python bridge needed at inference time.
- Override EP order via JVM property: `-Djforge.ep=cpu|coreml|cuda|tensorrt|directml|openvino|rocm`
- Task tabs show preview images when output is generated, with **Open Output** to launch the file.
- If a model requires external tensor files (e.g. `weights.pb`), import the complete ONNX bundle into the model directory.
