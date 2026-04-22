# ENGINE_DESIGN.md

## Purpose
Define the internal architecture of the JForge inference engine so the codebase can be refactored into a clean, modular, reproducible system.

## 1. Engine Overview

The engine is composed of:

- Task System (txt2img, img2img, inpaint, upscale)
- Model Registry
- Session Manager (ONNX Runtime)
- Schedulers
- Samplers
- Tokenizer + VAE + UNet pipelines
- Per-step callback system
- Metadata + output writer

## 2. Task System

### 2.1 Task Interface

```java
public interface ForgeTask {
    String id();
    TaskConfig config();
    TaskResult run(TaskContext ctx);
}
```

### 2.2 Task Implementations

- Txt2ImgTask
- Img2ImgTask
- InpaintTask
- UpscaleTask

Each task:

- Validates config
- Loads required models
- Creates inference pipeline
- Runs sampler
- Writes output + metadata

## 3. Session Manager

### 3.1 Responsibilities

- Create ONNX Runtime sessions
- Manage EP selection (CPU, CUDA, DirectML, CoreML)
- Cache sessions
- Track memory usage
- Provide thread-safe inference calls

### 3.2 API

```java
public class SessionManager {
    ORTSession getSession(ForgeModel model);
    void clearCache();
    DeviceInfo getActiveDevice();
}
```

## 4. Scheduler System

### 4.1 Interface

```java
public interface Scheduler {
    FloatTensor step(FloatTensor latents, int step, SchedulerState state);
}
```

### 4.2 Implementations

- Euler
- Euler A
- DPM++ 2M
- DPM++ SDE
- Flow Matching (SD3.5)

## 5. Sampler System

### 5.1 Interface

```java
public interface Sampler {
    FloatTensor sample(SamplerContext ctx);
}
```

### 5.2 Responsibilities

- Loop over steps
- Call scheduler
- Call UNet
- Apply guidance
- Emit per-step callback

## 6. Tokenizer + VAE + UNet

Implementation target layout:

- core/tokenizer for tokenizer logic
- core/vae for VAE encode/decode
- core/unet for UNet orchestration

## 7. Output Writer

Each generation produces:

```
output/
  YYYY-MM-DD/
    hh-mm-ss/
      image.png
      metadata.json
```

Metadata includes:

- Prompt
- Negative prompt
- Seed
- Steps
- CFG
- Sampler
- Scheduler
- Model ID
- Device
- Timing
