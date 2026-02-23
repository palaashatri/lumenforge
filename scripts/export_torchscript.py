#!/usr/bin/env python3
"""
Export Stable Diffusion v1.5 components to TorchScript format for LumenForge's
DJL/PyTorch backend.

Usage:
    pip install torch diffusers transformers accelerate
    python scripts/export_torchscript.py [--model_id MODEL] [--output_dir DIR]

Defaults:
    --model_id   stabilityai/stable-diffusion-2-1  (or runwayml/stable-diffusion-v1-5)
    --output_dir ~/.lumenforge-models/text-image/sd-pytorch

Produces:
    clip_model.pt          — traced CLIP text encoder
    unet_model.pt          — traced UNet2DConditionModel
    vae_decoder_model.pt   — traced VAE decoder
    tokenizer.json         — HuggingFace fast tokenizer config
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer


def main():
    parser = argparse.ArgumentParser(description="Export SD to TorchScript for LumenForge")
    parser.add_argument(
        "--model_id",
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model ID (default: runwayml/stable-diffusion-v1-5)"
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.expanduser("~/.lumenforge-models/text-image/sd-pytorch"),
        help="Output directory for TorchScript files"
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Model precision (float32 recommended for CPU, float16 for GPU)"
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print(f"🔧 Model: {args.model_id}")
    print(f"📁 Output: {output}")
    print(f"🎯 Precision: {args.dtype}")
    print()

    # ── Tokenizer ────────────────────────────────────────────────────
    print("1/4  Saving tokenizer…")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    tokenizer.save_pretrained(str(output))
    print(f"     ✓ tokenizer.json → {output / 'tokenizer.json'}")
    print()

    # ── CLIP text encoder ────────────────────────────────────────────
    print("2/4  Tracing CLIP text encoder…")
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id,
        subfolder="text_encoder",
        torchscript=True,
        torch_dtype=dtype,
        return_dict=False,
    ).eval()

    dummy_input = tokenizer(
        "a beautiful sunset over mountains",
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        traced_clip = torch.jit.trace(
            text_encoder,
            (dummy_input["input_ids"],),
        )
    clip_path = output / "clip_model.pt"
    torch.jit.save(traced_clip, str(clip_path))
    size_mb = clip_path.stat().st_size / 1024 / 1024
    print(f"     ✓ clip_model.pt ({size_mb:.1f} MB)")
    del text_encoder, traced_clip
    print()

    # ── UNet ─────────────────────────────────────────────────────────
    print("3/4  Tracing UNet (this may take a few minutes)…")
    unet = UNet2DConditionModel.from_pretrained(
        args.model_id,
        subfolder="unet",
        torchscript=True,
        torch_dtype=dtype,
        return_dict=False,
    ).eval()

    # Dummy inputs matching the UNet's expected shapes
    latent = torch.randn(1, 4, 64, 64, dtype=dtype)
    timestep = torch.tensor([981], dtype=torch.long)
    encoder_hidden_states = torch.randn(1, 77, 768, dtype=dtype)  # SD v1.5 dim

    with torch.no_grad():
        traced_unet = torch.jit.trace(
            unet,
            (latent, timestep, encoder_hidden_states),
        )
    unet_path = output / "unet_model.pt"
    torch.jit.save(traced_unet, str(unet_path))
    size_mb = unet_path.stat().st_size / 1024 / 1024
    print(f"     ✓ unet_model.pt ({size_mb:.1f} MB)")
    del unet, traced_unet
    print()

    # ── VAE decoder ──────────────────────────────────────────────────
    print("4/4  Tracing VAE decoder…")
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=dtype,
        return_dict=False,
    ).eval()

    # Only trace the decode path
    dummy_latent = torch.randn(1, 4, 64, 64, dtype=dtype)
    with torch.no_grad():
        traced_vae = torch.jit.trace(vae.decode, (dummy_latent,))
    vae_path = output / "vae_decoder_model.pt"
    torch.jit.save(traced_vae, str(vae_path))
    size_mb = vae_path.stat().st_size / 1024 / 1024
    print(f"     ✓ vae_decoder_model.pt ({size_mb:.1f} MB)")
    del vae, traced_vae
    print()

    # ── Summary ──────────────────────────────────────────────────────
    total = sum(f.stat().st_size for f in output.glob("*.pt")) / 1024 / 1024
    print("=" * 50)
    print(f"✅ Export complete! Total model size: {total:.0f} MB")
    print(f"   Output directory: {output}")
    print()
    print("Next steps:")
    print("  1. Build LumenForge with DJL:  mvn compile -Ddjl=true")
    print("  2. Run:                        mvn exec:java -Ddjl=true")
    print('  3. Select "SD v1.5 PyTorch (DJL)" from the model dropdown')
    print()


if __name__ == "__main__":
    main()
