# MODEL_REGISTRY.md

## Purpose
Define how JForge discovers, loads, validates, and manages models.

## 1. Model Definition

```java
public record ForgeModel(
    String id,
    Path path,
    ModelType type,
    Map<String, Object> metadata
) {}
```

## 2. Model Types

- SD15
- SDXL
- SD3_5
- ESRGAN
- CUSTOM

## 3. Model Metadata Format

Each model has a model.json:

```json
{
  "id": "sdxl-base",
  "type": "SDXL",
  "version": "1.0",
  "tags": ["realistic", "base"],
  "files": {
    "unet": "unet.onnx",
    "vae": "vae.onnx",
    "tokenizer": "tokenizer.json"
  },
  "requirements": {
    "minVramGB": 6,
    "recommendedVramGB": 12
  }
}
```

## 4. Model Registry Responsibilities

- Discover models in:
  - ~/.jforge/models
  - ./models
- Validate required files
- Load metadata
- Detect model type
- Provide model list to UI
- Provide compatibility checks

## 5. Model Compatibility Checks

- VRAM check
- Required components check
- Scheduler compatibility
- Sampler compatibility
- SD3.5 flow-matching requirement

## 6. Auto-Conversion (Existing Feature)

JForge supports:

- PyTorch -> ONNX conversion
- HuggingFace model download

This must remain preserved.

## 7. Model Selection Logic

When user selects a model:

1. Registry returns ForgeModel
2. SessionManager loads sessions
3. UI updates available samplers/schedulers
4. Engine uses correct pipeline
