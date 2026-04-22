# **GEMINI.MD — JForge Transformation Specification**
**Purpose:**  
Modify the existing JForge application (Java + Swing + ONNX Runtime) into a **DiffusionBee‑style**, **cross‑platform**, **offline**, **production‑grade** Stable Diffusion desktop app with a modern UI, modular architecture, and clean separation between UI, engine, and model management.

Gemini must **modify the existing codebase**, not rewrite from scratch.

---

## **1. High‑Level Goals**
Transform JForge into:

1. A **single‑window**, modern, minimal UI similar to DiffusionBee:
   - Left: Prompt panel  
   - Center: Output preview  
   - Right: Settings panel  
   - Bottom: History strip

2. A **modular task system**:
   - txt2img  
   - img2img  
   - inpainting  
   - upscaling  
   - (future) text‑to‑video

3. A **clean engine layer**:
   - Pure Java orchestrator  
   - JNI → ONNX Runtime  
   - Unified scheduler + sampler interface  
   - Unified model registry

4. A **cross‑platform packaging pipeline**:
   - macOS `.app` bundle  
   - Windows `.exe` launcher  
   - Linux `.AppImage` or `.sh`

5. A **strict, reproducible architecture**:
   - No improvisation  
   - No hidden state  
   - All generation configs stored as JSON  
   - All outputs stored with metadata

---

## **2. Required Architectural Changes**
Gemini must apply these modifications to the existing repo.

### **2.1 Create a new module layout**
```
jforge/
  core/               # inference engine, schedulers, model registry
  ui/                 # Swing UI (later replaceable with Compose)
  models/             # model metadata + discovery
  tasks/              # txt2img, img2img, inpaint, upscale
  runtime/            # EP selection, session mgmt, caching
  assets/             # icons, themes
```

### **2.2 Replace the current Swing UI with a modern layout**
Gemini must:

- Keep Swing (for now), but:
  - Replace JFrame layout with a **three‑pane layout**  
  - Add a **history strip**  
  - Add a **task selector** (tabs on top)  
  - Add a **live progress bar** with ETA  
  - Add **image preview** with zoom + pan  
  - Add **dark/light theme** via FlatLaf

### **2.3 Introduce a unified Task API**
Create:

```java
public interface ForgeTask {
    String id();
    TaskConfig config();
    TaskResult run(TaskContext ctx);
}
```

Implementations:

- `Txt2ImgTask`
- `Img2ImgTask`
- `InpaintTask`
- `UpscaleTask`

### **2.4 Introduce a unified Model API**
Gemini must create:

```java
public record ForgeModel(
    String id,
    Path path,
    ModelType type,
    Map<String, Object> metadata
) {}
```

Model types:

- SD15
- SDXL
- SD3.5
- ESRGAN

### **2.5 Rewrite the inference pipeline into a clean orchestrator**
Gemini must:

- Move all ONNX Runtime session creation into `runtime/SessionManager`
- Add:
  - tokenizer cache  
  - scheduler interface  
  - sampler interface  
  - per‑step callback  
  - memory guard (max VRAM/CPU RAM)

---

## **3. UI Requirements**
Gemini must implement:

### **3.1 Main Window Layout**
```
+-------------------------------------------------------------+
|  Task Tabs: [Text→Image] [Image→Image] [Inpaint] [Upscale] |
+----------------------+----------------------+----------------+
| Prompt Panel         | Output Preview       | Settings       |
| (left)               | (center)             | (right)        |
+-------------------------------------------------------------+
| History Strip (bottom)                                     |
+-------------------------------------------------------------+
```

### **3.2 Prompt Panel**
- Prompt  
- Negative prompt  
- Seed  
- Batch count  
- Style presets  
- Aspect ratio presets  

### **3.3 Settings Panel**
- Steps  
- CFG  
- Sampler  
- Scheduler  
- Model selector  
- Device selector (CPU/GPU/EP override)

### **3.4 Output Preview**
- Zoom  
- Pan  
- Save  
- Open folder  
- Metadata popup  

### **3.5 History Strip**
- Thumbnails  
- Click to load metadata  
- Right‑click → open folder  

---

## **4. Engine Requirements**
Gemini must:

### **4.1 Implement a per‑step callback**
```java
public interface StepCallback {
    void onStep(int step, float progress, long elapsedMs);
}
```

### **4.2 Implement a unified scheduler interface**
```java
public interface Scheduler {
    FloatTensor step(FloatTensor latents, int step, SchedulerState state);
}
```

### **4.3 Implement a unified sampler interface**
```java
public interface Sampler {
    FloatTensor sample(SamplerContext ctx);
}
```

### **4.4 Add support for:**
- Euler  
- Euler A  
- DPM++ 2M  
- DPM++ SDE  
- Flow Matching (SD3.5)

---

## **5. Model Management**
Gemini must:

- Keep the existing HuggingFace auto‑discovery  
- Keep PyTorch → ONNX auto‑conversion  
- Add:
  - Model metadata JSON  
  - Model versioning  
  - Model tags (e.g., “anime”, “realistic”, “turbo”)  
  - Model compatibility checks  

---

## **6. Packaging Requirements**
Gemini must create:

### **6.1 macOS**
- `.app` bundle  
- Native icon  
- Gatekeeper‑safe structure  

### **6.2 Windows**
- `.exe` launcher  
- Embedded JRE  
- Icon + version metadata  

### **6.3 Linux**
- `.AppImage`  
- Embedded JRE  

---

## **7. Non‑Negotiable Rules for Gemini**
Gemini must follow these rules strictly:

1. **Modify the existing codebase**, do not rewrite.  
2. **Preserve all existing ONNX Runtime logic.**  
3. **Preserve all existing model support (SD15 → SD3.5).**  
4. **Preserve all existing EP auto‑selection logic.**  
5. **Do not introduce Kotlin or Compose unless explicitly asked.**  
6. **Do not remove Swing unless explicitly asked.**  
7. **All new code must compile immediately.**  
8. **All new modules must be added to `pom.xml`.**  
9. **All new UI must be wired to real engine calls.**  
10. **No placeholders. No TODOs. No stubs.**

---

## **8. Deliverables Gemini Must Produce**
Gemini must output:

1. **A full patch** modifying the repo to match this spec.  
2. **New Java classes** for:
   - Task API  
   - Model API  
   - Scheduler  
   - Sampler  
   - SessionManager  
3. **New Swing UI** with the layout described.  
4. **Updated pom.xml** with module structure.  
5. **Updated README** describing the new app.  
6. **Build scripts** for macOS/Windows/Linux packaging.  

---

## **9. Final Instruction**
Gemini must treat this file as the **authoritative specification** for transforming JForge into the new application.  
Gemini must not deviate, simplify, or reinterpret the requirements.

---

Perfect — I’ll generate all **three** files you asked for:

- **UI_WIREFRAMES.md**  
- **ENGINE_DESIGN.md**  
- **MODEL_REGISTRY.md**

These are written specifically for **your existing JForge repo**, based on the structure visible in your active GitHub tab, and aligned with the transformation spec in **GEMINI.MD**.

They are ready to drop directly into the repo.

---

# **UI_WIREFRAMES.md**

## **Purpose**
Define the complete visual structure of the new JForge UI so Gemini (or Copilot) can implement it in Swing without ambiguity.

The UI follows a **DiffusionBee‑style**, **three‑pane**, **single‑window** layout with a bottom history strip.

---

## **1. Main Window Layout**

```
+--------------------------------------------------------------------------------+
|  [Text → Image] [Image → Image] [Inpaint] [Upscale] [Settings]                |
+----------------------+-------------------------------+-------------------------+
|                      |                               |                         |
|  Prompt Panel        |       Output Preview          |     Settings Panel      |
|  (Left Sidebar)      |       (Center Canvas)         |     (Right Sidebar)     |
|                      |                               |                         |
+----------------------+-------------------------------+-------------------------+
|                          Generation History Strip                           |
+--------------------------------------------------------------------------------+
```

---

## **2. Prompt Panel (Left Sidebar)**

```
+----------------------+
| Prompt               |
| [ multiline box ]    |
|                      |
| Negative Prompt      |
| [ multiline box ]    |
|                      |
| Seed [   12345   ]   |
| Batch Count [ 1 ]    |
|                      |
| Style Presets        |
|  - Photorealistic    |
|  - Anime             |
|  - Illustration      |
|  - Cinematic         |
|                      |
| Aspect Ratio         |
|  [1:1] [16:9] [9:16] |
+----------------------+
```

---

## **3. Output Preview (Center)**

```
+-----------------------------------------------+
| [Generated Image Canvas]                      |
|                                               |
|  - Zoom slider                                |
|  - Pan with mouse drag                        |
|  - Right-click: "Open Folder", "Show Metadata"|
+-----------------------------------------------+
```

---

## **4. Settings Panel (Right Sidebar)**

```
+----------------------+
| Model Selector       |
| [ dropdown ]         |
|                      |
| Device               |
| [ CPU | GPU | EP ]   |
|                      |
| Steps [   30   ]     |
| CFG   [   7.5  ]     |
| Sampler              |
| [ Euler | DPM++ ]    |
| Scheduler            |
| [ DDIM | Flow ]      |
|                      |
| Resolution           |
| [512x512] [768x768]  |
| Custom: [ W ] [ H ]  |
+----------------------+
```

---

## **5. History Strip (Bottom)**

```
+----------------------------------------------------------------+
| [thumb][thumb][thumb][thumb][thumb][thumb][thumb][thumb]      |
|  click → load metadata                                         |
|  right-click → open folder                                     |
+----------------------------------------------------------------+
```

---

# **ENGINE_DESIGN.md**

## **Purpose**
Define the internal architecture of the JForge inference engine so Gemini can refactor the existing code into a clean, modular, reproducible system.

---

# **1. Engine Overview**

The engine is composed of:

- **Task System** (txt2img, img2img, inpaint, upscale)
- **Model Registry**
- **Session Manager** (ONNX Runtime)
- **Schedulers**
- **Samplers**
- **Tokenizer + VAE + UNet pipelines**
- **Per‑step callback system**
- **Metadata + output writer**

---

# **2. Task System**

## **2.1 Task Interface**

```java
public interface ForgeTask {
    String id();
    TaskConfig config();
    TaskResult run(TaskContext ctx);
}
```

## **2.2 Task Implementations**

- `Txt2ImgTask`
- `Img2ImgTask`
- `InpaintTask`
- `UpscaleTask`

Each task:

- Validates config  
- Loads required models  
- Creates inference pipeline  
- Runs sampler  
- Writes output + metadata  

---

# **3. Session Manager**

## **3.1 Responsibilities**

- Create ONNX Runtime sessions  
- Manage EP selection (CPU, CUDA, DirectML, CoreML)  
- Cache sessions  
- Track memory usage  
- Provide thread‑safe inference calls  

## **3.2 API**

```java
public class SessionManager {
    ORTSession getSession(ForgeModel model);
    void clearCache();
    DeviceInfo getActiveDevice();
}
```

---

# **4. Scheduler System**

## **4.1 Interface**

```java
public interface Scheduler {
    FloatTensor step(FloatTensor latents, int step, SchedulerState state);
}
```

## **4.2 Implementations**

- Euler  
- Euler A  
- DPM++ 2M  
- DPM++ SDE  
- Flow Matching (SD3.5)

---

# **5. Sampler System**

## **5.1 Interface**

```java
public interface Sampler {
    FloatTensor sample(SamplerContext ctx);
}
```

## **5.2 Responsibilities**

- Loop over steps  
- Call scheduler  
- Call UNet  
- Apply guidance  
- Emit per‑step callback  

---

# **6. Tokenizer + VAE + UNet**

Gemini must:

- Move all tokenizer logic into `core/tokenizer`  
- Move VAE encode/decode into `core/vae`  
- Move UNet into `core/unet`  

---

# **7. Output Writer**

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

---

# **MODEL_REGISTRY.md**

## **Purpose**
Define how JForge discovers, loads, validates, and manages models.

---

# **1. Model Definition**

```java
public record ForgeModel(
    String id,
    Path path,
    ModelType type,
    Map<String, Object> metadata
) {}
```

---

# **2. Model Types**

- `SD15`
- `SDXL`
- `SD3_5`
- `ESRGAN`
- `CUSTOM`

---

# **3. Model Metadata Format**

Each model has a `model.json`:

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

---

# **4. Model Registry Responsibilities**

- Discover models in:
  - `~/.jforge/models`
  - `./models`
- Validate required files  
- Load metadata  
- Detect model type  
- Provide model list to UI  
- Provide compatibility checks  

---

# **5. Model Compatibility Checks**

Gemini must implement:

- VRAM check  
- Required components check  
- Scheduler compatibility  
- Sampler compatibility  
- SD3.5 flow‑matching requirement  

---

# **6. Auto‑Conversion (Existing Feature)**

JForge already supports:

- PyTorch → ONNX conversion  
- HuggingFace model download  

Gemini must **preserve** this.

---

# **7. Model Selection Logic**

When user selects a model:

1. Registry returns `ForgeModel`  
2. SessionManager loads sessions  
3. UI updates available samplers/schedulers  
4. Engine uses correct pipeline  

---