# UI_WIREFRAMES.md

## Purpose
Define the complete visual structure of the new JForge UI so implementation in Swing is unambiguous.

The UI follows a DiffusionBee-style, three-pane, single-window layout with a bottom history strip.

## 1. Main Window Layout

```
+--------------------------------------------------------------------------------+
|  [Text -> Image] [Image -> Image] [Inpaint] [Upscale] [Settings]              |
+----------------------+-------------------------------+-------------------------+
|                      |                               |                         |
|  Prompt Panel        |       Output Preview          |     Settings Panel      |
|  (Left Sidebar)      |       (Center Canvas)         |     (Right Sidebar)     |
|                      |                               |                         |
+----------------------+-------------------------------+-------------------------+
|                          Generation History Strip                              |
+--------------------------------------------------------------------------------+
```

## 2. Prompt Panel (Left Sidebar)

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

## 3. Output Preview (Center)

```
+-----------------------------------------------+
| [Generated Image Canvas]                      |
|                                               |
|  - Zoom slider                                |
|  - Pan with mouse drag                        |
|  - Right-click: Open Folder, Show Metadata    |
+-----------------------------------------------+
```

## 4. Settings Panel (Right Sidebar)

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

## 5. History Strip (Bottom)

```
+----------------------------------------------------------------+
| [thumb][thumb][thumb][thumb][thumb][thumb][thumb][thumb]      |
|  click -> load metadata                                         |
|  right-click -> open folder                                     |
+----------------------------------------------------------------+
```
