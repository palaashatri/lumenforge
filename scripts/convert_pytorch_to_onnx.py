#!/usr/bin/env python3
"""
PyTorch → ONNX converter for LumenForge.

Converts HuggingFace diffusers pipelines (Stable Diffusion, SDXL, etc.)
or generic PyTorch models (.pt / .pth) to ONNX format for inference
with ONNX Runtime.

Usage — diffusers pipeline (recommended for SD models):
    python convert_pytorch_to_onnx.py \
        --model_id runwayml/stable-diffusion-v1-5 \
        --output_dir ~/.lumenforge-models/text-image/converted/sd-v15

Usage — generic PyTorch model (per Microsoft tutorial):
    python convert_pytorch_to_onnx.py \
        --model_id ./my_model.pt \
        --output_dir ./output \
        --mode generic \
        --input_shape 1,3,224,224

References:
    https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-convert-model
    https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model
"""

import argparse
import os
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


# ── Diffusers pipeline export (optimum) ──────────────────────────────────

def convert_diffusers(model_id, output_dir):
    """
    Export a HuggingFace diffusers pipeline to ONNX using the Optimum library.

    Produces separate ONNX files for each component:
        unet/model.onnx or transformer/model.onnx, text_encoder/model.onnx,
        vae_decoder/model.onnx, etc.
    """
    progress(5, "Importing optimum exporters…")
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        error(
            "Required package 'optimum[exporters]' is not installed.\n"
            "Install with:  pip install 'optimum[exporters]'"
        )

    progress(10, f"Resolving model: {model_id}")

    # Try common diffusers export tasks in order of specificity.
    # SD 3.x uses 'stable-diffusion-3', SDXL uses 'stable-diffusion-xl',
    # SD 1.x/2.x uses 'stable-diffusion'.
    tasks = ["stable-diffusion-3", "stable-diffusion-xl", "stable-diffusion"]

    for task in tasks:
        try:
            progress(15, f"Attempting export with task='{task}'…")
            # For SD 3.x, skip T5 by default to keep model size manageable
            kwargs = {
                "model_name_or_path": model_id,
                "output": Path(output_dir),
                "task": task,
                "fp16": False,
            }
            # Some optimum versions accept no_post_process for SD3
            if task == "stable-diffusion-3":
                progress(15, "Note: SD 3.x export may take 10-30 minutes and ~32GB RAM.")
            main_export(**kwargs)
            progress(100, f"Export complete (task={task}).")
            return
        except Exception as exc:
            msg = str(exc).lower()
            # If the error is about wrong task type, try the next one
            if "task" in msg or "not supported" in msg or "auto" in msg:
                progress(15, f"Task '{task}' not compatible — trying next…")
                continue
            # Real error — report and stop
            error(str(exc))

    error(f"Could not auto-detect a compatible export task for '{model_id}'. "
          f"Tried: {', '.join(tasks)}")


# ── Generic PyTorch model export (torch.onnx.export) ────────────────────

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
        error("Required package 'torch' is not installed.\nInstall with:  pip install torch")

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


# ── CLI entry point ─────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="LumenForge PyTorch → ONNX converter"
    )
    ap.add_argument(
        "--model_id", required=True,
        help="HuggingFace model ID (e.g. 'runwayml/stable-diffusion-v1-5') "
             "or local path to a .pt/.pth file",
    )
    ap.add_argument(
        "--output_dir", required=True,
        help="Directory to save converted ONNX model(s)",
    )
    ap.add_argument(
        "--mode", default="diffusers",
        choices=["diffusers", "generic"],
        help="'diffusers' for HuggingFace pipeline models, "
             "'generic' for standalone .pt/.pth files (default: diffusers)",
    )
    ap.add_argument(
        "--input_shape", default="1,3,224,224",
        help="Input tensor shape for generic mode, comma-separated "
             "(default: 1,3,224,224)",
    )
    ap.add_argument(
        "--hf_token", default="",
        help="HuggingFace auth token for gated models (e.g. SD 3.5)",
    )
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.hf_token:
        setup_hf_token(args.hf_token)

    if args.mode == "diffusers":
        convert_diffusers(args.model_id, args.output_dir)
    else:
        convert_generic(args.model_id, args.output_dir, args.input_shape)


if __name__ == "__main__":
    main()
