#!/usr/bin/env python3
"""
PyTorch → ONNX converter for JForge.

Converts HuggingFace diffusers pipelines (Stable Diffusion, SDXL,
text-to-video, etc.) or generic PyTorch models (.pt / .pth) to ONNX
format for inference with ONNX Runtime.

Usage — diffusers pipeline (recommended for SD / video models):
    python convert_pytorch_to_onnx.py \\
        --model_id ali-vilab/text-to-video-ms-1.7b \\
        --output_dir ~/.jforge-models/text-video/converted/modelscope-t2v

    python convert_pytorch_to_onnx.py \\
        --model_id runwayml/stable-diffusion-v1-5 \\
        --output_dir ~/.jforge-models/text-image/converted/sd-v15

Usage — generic PyTorch model (per Microsoft tutorial):
    python convert_pytorch_to_onnx.py \\
        --model_id ./my_model.pt \\
        --output_dir ./output \\
        --mode generic \\
        --input_shape 1,3,224,224

Usage — force-reinstall optimum if imports are broken:
    python convert_pytorch_to_onnx.py --reinstall \\
        --model_id ali-vilab/text-to-video-ms-1.7b \\
        --output_dir /tmp/test-out

References:
    https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model
    https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model
    https://huggingface.co/ali-vilab/text-to-video-ms-1.7b
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def progress(pct, msg):
    """Print a structured progress line that Java can parse."""
    print(f"PROGRESS: {pct} {msg}", flush=True)


def error(msg):
    """Print a structured error line and exit."""
    print(f"ERROR: {msg}", flush=True)
    sys.exit(1)


def setup_hf_token(token):
    """Set HuggingFace auth token for gated model access."""
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            progress(3, "Authenticated with Hugging Face.")
        except ImportError:
            progress(3, "HF token set via environment variable.")


def _get_optimum_version():
    """Return the installed optimum version string, or 'unknown'."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "optimum"],
            capture_output=True, text=True, timeout=15
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def _get_optimum_location():
    """Return the installed optimum Location, or 'unknown'."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "optimum"],
            capture_output=True, text=True, timeout=15
        )
        for line in result.stdout.splitlines():
            if line.startswith("Location:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def reinstall_optimum():
    """Force-reinstall optimum[exporters] to fix broken installs."""
    progress(2, "Force-reinstalling optimum[exporters]…")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--force-reinstall", "--no-cache-dir",
            "optimum[exporters]>=1.19.0"
        ])
        progress(4, "optimum[exporters] reinstalled successfully.")
    except subprocess.CalledProcessError as e:
        error(f"Failed to reinstall optimum[exporters]: {e}\n"
              f"Run manually: pip install --force-reinstall 'optimum[exporters]>=1.19.0'")


def import_main_export():
    """
    Import optimum's main_export function using a multi-strategy approach
    to handle different versions and broken installs.

    Strategy order:
      1. optimum.exporters.onnx.main_export  (optimum >= 1.14, current)
      2. optimum.onnx.main_export            (older pre-1.14 API)

    Returns the callable on success, or calls error() and exits.
    """
    # Strategy 1: current API (optimum >= 1.14)
    try:
        from optimum.exporters.onnx import main_export
        return main_export
    except ImportError as e1:
        first_error = str(e1)

    # Strategy 2: older API fallback
    try:
        from optimum.onnx import main_export  # noqa: F811
        progress(5, "Note: using legacy optimum.onnx API — consider upgrading optimum.")
        return main_export
    except ImportError as e2:
        second_error = str(e2)

    # Both failed — provide a maximally actionable error message
    ver = _get_optimum_version()
    loc = _get_optimum_location()
    error(
        f"Could not import optimum's ONNX exporter.\n"
        f"  Installed optimum version : {ver}\n"
        f"  optimum location          : {loc}\n"
        f"  optimum.exporters.onnx    : {first_error}\n"
        f"  optimum.onnx              : {second_error}\n"
        f"\n"
        f"Fix with one of:\n"
        f"  pip install --upgrade 'optimum[exporters]>=1.19.0'\n"
        f"  pip install --force-reinstall --no-cache-dir 'optimum[exporters]>=1.19.0'\n"
        f"\n"
        f"Or re-run this script with --reinstall to do it automatically."
    )


# ── Diffusers pipeline export (optimum) ──────────────────────────────────────

def _is_text_to_video_model(model_id: str) -> bool:
    """Heuristic: detect models that need custom component export instead of optimum."""
    lower = model_id.lower()
    return any(k in lower for k in [
        "text-to-video", "t2v", "vilab", "modelscope", "cogvideo", "videocrafter",
        "animatediff", "zeroscope"
    ])


def convert_diffusers_custom_t2v(model_id: str, output_dir: str) -> None:
    """
    Export a text-to-video diffusers pipeline to ONNX by tracing each
    component individually with torch.onnx.export.

    This is required for models like ali-vilab/text-to-video-ms-1.7b whose
    UNet3D architecture is not registered in optimum's task registry.

    Outputs (relative to output_dir):
        text_encoder/model.onnx
        unet/model.onnx            (the temporal UNet3D)
        vae_decoder/model.onnx
    """
    progress(5, "Importing PyTorch + diffusers…")
    try:
        import torch
        import torch.onnx
    except ImportError:
        error("torch is not installed. Run: pip install -r scripts/requirements.txt")

    try:
        from diffusers import DiffusionPipeline
    except ImportError:
        error("diffusers is not installed. Run: pip install -r scripts/requirements.txt")

    out = Path(output_dir)
    device = "cpu"  # ONNX Runtime on macOS/CPU only; move to cuda if available
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        device = "cuda"

    progress(10, f"Loading pipeline: {model_id}  (device={device})")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,   # ModelScope pipelines need this
        )
        pipe = pipe.to(device)
    except Exception as exc:
        error(f"Failed to load pipeline '{model_id}': {exc}")

    # ── 1. Text Encoder ──────────────────────────────────────────────────────
    progress(20, "Exporting text encoder to ONNX…")
    try:
        text_encoder = pipe.text_encoder.eval()
        te_out = out / "text_encoder"
        te_out.mkdir(parents=True, exist_ok=True)

        # Typical CLIP text encoder: input_ids [batch, seq_len=77]
        dummy_ids = torch.zeros(1, 77, dtype=torch.long, device=device)
        with torch.no_grad():
            torch.onnx.export(
                text_encoder,
                (dummy_ids,),
                str(te_out / "model.onnx"),
                opset_version=17,
                input_names=["input_ids"],
                output_names=["last_hidden_state", "pooler_output"],
                dynamic_axes={"input_ids": {0: "batch_size"}},
                do_constant_folding=True,
            )
        progress(30, "Text encoder exported.")
    except Exception as exc:
        progress(25, f"Warning: text encoder export failed ({exc}) — skipping.")

    # ── 2. UNet (3D temporal UNet / spatial-temporal denoiser) ───────────────
    progress(35, "Exporting UNet3D to ONNX…")
    try:
        unet = pipe.unet.eval() if hasattr(pipe, "unet") else None
        if unet is not None:
            unet_out = out / "unet"
            unet_out.mkdir(parents=True, exist_ok=True)

            # Probe UNet config for shape hints
            in_ch = getattr(unet.config, "in_channels", 4)
            # For text-to-video-ms-1.7b: [batch, channels, frames, H, W]
            # frames=16 is the default for this model at 256p
            batch, frames, h, w = 1, 16, 32, 32  # latent spatial (256/8=32)
            seq_len = 77
            cross_attn_dim = getattr(unet.config, "cross_attention_dim", 1024)

            dummy_latent    = torch.randn(batch, in_ch, frames, h, w, device=device)
            dummy_timestep  = torch.tensor([1], dtype=torch.long, device=device)
            dummy_encoder_hs = torch.randn(batch, seq_len, cross_attn_dim, device=device)

            with torch.no_grad():
                torch.onnx.export(
                    unet,
                    (dummy_latent, dummy_timestep, dummy_encoder_hs),
                    str(unet_out / "model.onnx"),
                    opset_version=17,
                    input_names=["sample", "timestep", "encoder_hidden_states"],
                    output_names=["out_sample"],
                    dynamic_axes={
                        "sample":                {0: "batch_size", 2: "num_frames"},
                        "encoder_hidden_states": {0: "batch_size"},
                    },
                    do_constant_folding=True,
                )
            progress(60, "UNet3D exported.")
        else:
            progress(40, "Warning: no 'unet' attribute on pipeline — UNet export skipped.")
    except Exception as exc:
        progress(40, f"Warning: UNet export failed ({exc}) — skipping.")

    # ── 3. VAE Decoder ───────────────────────────────────────────────────────
    progress(65, "Exporting VAE decoder to ONNX…")
    try:
        vae = pipe.vae.eval() if hasattr(pipe, "vae") else None
        if vae is not None:
            vae_out = out / "vae_decoder"
            vae_out.mkdir(parents=True, exist_ok=True)

            latent_ch = getattr(vae.config, "latent_channels", 4)
            dummy_latent_vae = torch.randn(1, latent_ch, h, w, device=device)

            with torch.no_grad():
                torch.onnx.export(
                    vae.decode,                  # export decode() method only
                    (dummy_latent_vae,),
                    str(vae_out / "model.onnx"),
                    opset_version=17,
                    input_names=["latent_sample"],
                    output_names=["sample"],
                    dynamic_axes={"latent_sample": {0: "batch_size"}},
                    do_constant_folding=True,
                )
            progress(85, "VAE decoder exported.")
        else:
            progress(70, "Warning: no 'vae' attribute on pipeline — VAE export skipped.")
    except Exception as exc:
        progress(70, f"Warning: VAE decoder export failed ({exc}) — skipping.")

    # ── 4. Tokenizer config (for Java-side ClipTokenizer) ────────────────────
    try:
        if hasattr(pipe, "tokenizer") and pipe.tokenizer is not None:
            tok_out = out / "tokenizer"
            tok_out.mkdir(parents=True, exist_ok=True)
            pipe.tokenizer.save_pretrained(str(tok_out))
            progress(90, "Tokenizer saved.")
    except Exception as exc:
        progress(88, f"Warning: tokenizer save failed ({exc}) — continuing.")

    progress(100, f"Text-to-video component export complete → {output_dir}")


# ── Diffusers pipeline export (optimum) ──────────────────────────────────────

def convert_diffusers(model_id, output_dir):
    """
    Export a HuggingFace diffusers pipeline to ONNX.

    - For text-to-video models (ali-vilab/text-to-video-ms-1.7b, etc.):
      Uses custom per-component torch.onnx.export because the UNet3D
      architecture is not in optimum's task registry.
    - For all other models (SD, SDXL, SD3, etc.):
      Uses optimum.exporters.onnx.main_export with auto task detection.
    """
    if _is_text_to_video_model(model_id):
        progress(5, f"Detected text-to-video model — using custom component export for '{model_id}'")
        convert_diffusers_custom_t2v(model_id, output_dir)
        return

    progress(5, "Importing optimum exporters…")
    main_export = import_main_export()

    progress(10, f"Resolving model: {model_id}")

    # Try common diffusers export tasks in order of specificity.
    # "auto" lets optimum detect the task from the model's config.json.
    tasks = [
        "auto",
        "stable-diffusion-3",
        "stable-diffusion-xl",
        "stable-diffusion",
    ]

    last_exc = None
    for task in tasks:
        try:
            progress(15, f"Attempting export with task='{task}'…")
            kwargs = {
                "model_name_or_path": model_id,
                "output": Path(output_dir),
                "task": task,
                "fp16": False,
            }
            if task == "stable-diffusion-3":
                progress(15, "Note: SD 3.x export may take 10-30 minutes and ~32GB RAM.")
            main_export(**kwargs)
            progress(100, f"Export complete (task={task}).")
            return
        except Exception as exc:
            last_exc = exc
            progress(15, f"Task '{task}' failed: {exc} — trying next…")
            continue

    error(
        f"Could not export '{model_id}' — all {len(tasks)} task strategies failed.\n"
        f"Last error: {last_exc}"
    )



# ── Generic PyTorch model export (torch.onnx.export) ─────────────────────────

def convert_generic(model_path, output_dir, input_shape_str):
    """
    Export a standalone .pt / .pth model using torch.onnx.export().

    This follows the approach recommended by Microsoft:
    https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model
    """
    progress(5, "Importing PyTorch…")
    try:
        import torch
        import torch.onnx
    except ImportError:
        error(
            "Required package 'torch' is not installed.\n"
            "Install with:  pip install torch\n"
            "Or:            pip install -r scripts/requirements.txt"
        )

    progress(15, f"Loading model: {model_path}")
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        error(f"Failed to load '{model_path}': {exc}")

    # Handle state-dict only checkpoints
    if isinstance(model, dict):
        if "model" in model and hasattr(model["model"], "forward"):
            model = model["model"]
        else:
            error(
                "File contains only a state_dict (weights without architecture).\n"
                "A full model with a forward() method is required for generic export.\n"
                "For HuggingFace / diffusers models, use --mode diffusers instead."
            )

    if not hasattr(model, "forward"):
        error("Loaded object does not have a forward() method — cannot export.")

    progress(25, "Setting model to eval mode…")
    model.eval()

    progress(35, "Creating dummy input tensor…")
    shape = [int(d.strip()) for d in input_shape_str.split(",")]
    dummy_input = torch.randn(*shape, requires_grad=False)

    out_path = Path(output_dir) / "model.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    progress(50, "Running torch.onnx.export()…")
    try:
        torch.onnx.export(
            model,                      # model being converted
            dummy_input,                # model input (or tuple for multiple)
            str(out_path),              # output file path
            export_params=True,         # store trained weights in the model file
            opset_version=17,           # ONNX opset version
            do_constant_folding=True,   # fold constants for optimization
            input_names=["modelInput"],
            output_names=["modelOutput"],
            dynamic_axes={
                "modelInput":  {0: "batch_size"},
                "modelOutput": {0: "batch_size"},
            },
        )
    except Exception as exc:
        error(f"torch.onnx.export() failed: {exc}")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    progress(85, f"Wrote {out_path.name} ({size_mb:.1f} MB)")

    # Quick ONNX validation
    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(out_path)))
        progress(95, "ONNX checker: model is valid ✓")
    except ImportError:
        progress(95, "Skipping ONNX validation (onnx package not installed)")
    except Exception as exc:
        progress(95, f"ONNX checker warning: {exc}")

    progress(100, "Conversion complete!")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="JForge PyTorch → ONNX converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Text-to-video (ModelScope)\n"
            "  %(prog)s --model_id ali-vilab/text-to-video-ms-1.7b --output_dir ~/models/t2v\n\n"
            "  # Stable Diffusion\n"
            "  %(prog)s --model_id runwayml/stable-diffusion-v1-5 --output_dir ~/models/sd\n\n"
            "  # Fix broken optimum install then convert\n"
            "  %(prog)s --reinstall --model_id ali-vilab/text-to-video-ms-1.7b --output_dir ~/models/t2v\n"
        )
    )
    ap.add_argument(
        "--model_id", required=True,
        help=(
            "HuggingFace model ID (e.g. 'ali-vilab/text-to-video-ms-1.7b') "
            "or local path to a .pt/.pth file"
        ),
    )
    ap.add_argument(
        "--output_dir", required=True,
        help="Directory to save converted ONNX model(s)",
    )
    ap.add_argument(
        "--mode", default="diffusers",
        choices=["diffusers", "generic"],
        help=(
            "'diffusers' for HuggingFace pipeline models (default), "
            "'generic' for standalone .pt/.pth files"
        ),
    )
    ap.add_argument(
        "--input_shape", default="1,3,224,224",
        help="Input tensor shape for generic mode, comma-separated (default: 1,3,224,224)",
    )
    ap.add_argument(
        "--hf_token", default="",
        help="HuggingFace auth token for gated models (e.g. SD 3.5)",
    )
    ap.add_argument(
        "--reinstall", action="store_true",
        help=(
            "Force-reinstall optimum[exporters] before importing. "
            "Fixes 'unknown location' and broken install errors."
        ),
    )
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.hf_token:
        setup_hf_token(args.hf_token)

    if args.reinstall:
        reinstall_optimum()

    if args.mode == "diffusers":
        convert_diffusers(args.model_id, args.output_dir)
    else:
        convert_generic(args.model_id, args.output_dir, args.input_shape)


if __name__ == "__main__":
    main()
