"""Image classification script using Cloudflare Clef multimodal decision model.

Model: https://huggingface.co/Cloudflare/clef
Default target image: 20250221_222154.jpg
Classes: human, cat, bike, car
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import torch
from huggingface_hub import snapshot_download

# Ensure local directory is in python path to import joint_schema_model
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Check for local HF_TOKEN file and load it safely
token_file = script_dir / "HF_TOKEN"
if token_file.is_file():
    token_val = token_file.read_text(encoding="utf-8").strip()
    if token_val:
        os.environ["HF_TOKEN"] = token_val
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token_val
        print("[*] Loaded Hugging Face token from 'HF_TOKEN' file.")

from joint_schema_model import (
    ClefModel,
    JointSchemaHead,
    collate_records,
    encode_record,
    load_release_model,
    systemone,
)


def load_clef(model_id_or_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load Clef model and processor, with support for GPU, CPU, and auto-offload.

    For 27B Cloudflare/clef on a 24GB GPU (RTX 4090), 'auto' automatically
    splits layers between GPU VRAM and CPU RAM.
    For 9B Cloudflare/clef-flash, 'cuda' fits entirely within 24GB VRAM.
    """
    path = Path(model_id_or_path)
    if not path.is_dir():
        print(f"[*] Downloading / loading repo '{model_id_or_path}' from Hugging Face...")
        path = Path(snapshot_download(model_id_or_path))

    print(f"[*] Loading model from: {path}")

    if device == "auto":
        from safetensors.torch import load_file
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        import json

        print("[*] Using device_map='auto' (balancing across GPU VRAM & CPU RAM)...")
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
            path,
            dtype=dtype,
            device_map="auto",
        )
        backbone.config.use_cache = False

        head_config = json.loads((path / "joint_head_config.json").read_text())
        head = JointSchemaHead(**head_config)
        head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
        # Place head on the device of the model's output embeddings
        target_device = backbone.get_output_embeddings().weight.device
        head = head.to(device=target_device, dtype=dtype)
        processor = AutoProcessor.from_pretrained(path)
        return ClefModel(backbone, head).eval(), processor
    else:
        return load_release_model(path, device=device, dtype=dtype)


def classify_image(
    image_path: str,
    model_id: str = "Cloudflare/clef",
    device: str = "cuda",
):
    """Classify whether the given image is a human, cat, bike, or car."""
    # 1. Validate image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'")

    print(f"[*] Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"    Dimensions: {image.size[0]}x{image.size[1]} px")

    # 2. Load model & processor
    model, processor = load_clef(model_id, device=device)

    # 3. Define the Clef Decision Schema
    # Clef supports choice (categorical selection) and noul (calibrated boolean probabilities)
    request = {
        "model": model_id,
        "state": {
            "task": "Classify the primary subject shown in the provided photograph."
        },
        "images": [image],
        "questions": {
            # Multi-class categorical question
            "subject_category": {
                "type": "choice",
                "instructions": "What is the primary subject in the photograph?",
                "criteria": {
                    "human": "A person, human being, man, woman, or child.",
                    "cat": "A cat, kitten, or feline animal.",
                    "bike": "A bicycle, bike, cyclist, or motorcycle.",
                    "car": "A car, automobile, or motor vehicle.",
                },
            },
            # Individual calibrated verification questions (true/false)
            "is_human": {
                "type": "noul",
                "instructions": "Is there a human or person in this photo?",
            },
            "is_cat": {
                "type": "noul",
                "instructions": "Is there a cat in this photo?",
            },
            "is_bike": {
                "type": "noul",
                "instructions": "Is there a bike or bicycle in this photo?",
            },
            "is_car": {
                "type": "noul",
                "instructions": "Is there a car or automobile in this photo?",
            },
        },
    }

    print("[*] Running inference via Clef decision head...")
    # Using SystemOne API helper built into joint_schema_model
    response = systemone(model, processor, request)

    # 4. Display Results
    answers = response["answers"]
    cat_answer = answers["subject_category"]
    winner = cat_answer["choice"]
    confidence = cat_answer["confidence"]
    probs = cat_answer["probabilities"]

    print("\n" + "=" * 55)
    print("           CLEF CLASSIFICATION RESULTS")
    print("=" * 55)
    print(f" Target Image : {image_path}")
    print(f" Model Used   : {model_id}")
    print(f" PREDICTION   : >> {winner.upper()} << (Confidence: {confidence * 100:.2f}%)")
    print("-" * 55)
    print(" Categorical Probabilities (Choice):")
    for category, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"   {category:<7} : {prob * 100:6.2f}% |{bar}|")

    print("-" * 55)
    print(" Independent Class Probabilities (Noul):")
    for q_id in ["is_human", "is_cat", "is_bike", "is_car"]:
        val = answers[q_id]["noul"]
        label = q_id.replace("is_", "").capitalize()
        bar_len = int(val * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"   {label:<7} : {val * 100:6.2f}% |{bar}|")
    print("=" * 55 + "\n")

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify image using Cloudflare Clef")
    parser.add_argument(
        "--image",
        type=str,
        default="20250221_222154.jpg",
        help="Path to the image file (default: 20250221_222154.jpg)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Cloudflare/clef",
        choices=["Cloudflare/clef", "Cloudflare/clef-flash"],
        help="Model ID: 'Cloudflare/clef' (27B) or 'Cloudflare/clef-flash' (9B, fast & fits 24GB VRAM)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "auto"],
        help="Device to use ('cuda', 'cpu', or 'auto' for offloading 27B across GPU+CPU)",
    )

    args = parser.parse_args()

    # Automatically recommend device='auto' if user chose 27B model on a single 24GB GPU
    device_to_use = args.device
    if args.model == "Cloudflare/clef" and device_to_use == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram_gb < 48:
            print(
                f"[!] Notice: 'Cloudflare/clef' (27B) requires ~55GB VRAM in bf16.\n"
                f"    Your GPU has {total_vram_gb:.1f} GB. Switching device to 'auto' (GPU + CPU offload)\n"
                f"    to avoid CUDA Out-of-Memory error. (Tip: Use '--model Cloudflare/clef-flash' to fit 100% in VRAM).\n"
            )
            device_to_use = "auto"

    classify_image(
        image_path=args.image,
        model_id=args.model,
        device=device_to_use,
    )
