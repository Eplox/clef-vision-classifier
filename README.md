# Image Classification with Cloudflare Clef

This project classifies images using the [Cloudflare Clef](https://huggingface.co/Cloudflare/clef) multimodal decision model family. Unlike generative vision-language models that output unstructured text, Clef evaluates typed schema questions and outputs calibrated probabilities for each allowed option in a single forward pass.

---

## Prerequisites

Ensure the required Python dependencies are installed in your environment:

```bash
pip install torch torchvision transformers huggingface_hub accelerate pillow
```

### Hugging Face Authentication
A valid token in the local file `HF_TOKEN` (or the `HF_TOKEN` environment variable) is automatically detected to allow high-speed, authenticated model downloads from Hugging Face Hub.

---

## Running the Script

### 1. Default Run (`Cloudflare/clef` - 27B)
Classifies the default image (`20250221_222154.jpg`) across `human`, `cat`, `bike`, and `car`:

```bash
python classify.py
```

> **Note on 27B & VRAM:**  
> `Cloudflare/clef` contains 27 billion parameters (~55 GB weights in `bfloat16`). On GPUs with 24 GB VRAM (such as an RTX 4090), the script automatically uses `device='auto'` to split layers across GPU VRAM and system RAM using `accelerate`.  
> *On the first run, Hugging Face will download the model weights to your local hub cache.*

---

### 2. Fast / Low-Latency Run (`Cloudflare/clef-flash` - 9B)
`Clef-Flash` has 9 billion parameters (~19 GB weights) and fits **100% directly inside 24 GB VRAM** for ultra-fast, zero-offload inference:

```bash
python classify.py --model Cloudflare/clef-flash
```

---

### 3. Classify a Custom Image
You can pass any image path using the `--image` flag:

```bash
python classify.py --image path/to/your/image.jpg

python classify.py --model Cloudflare/clef-flash --image 20250221_222154.jpg
```

---

### 4. Advanced Device Selection
Override device allocation manually (`cuda`, `auto`, or `cpu`):

```bash
# Force GPU + CPU offloading
python classify.py --device auto

# Run entirely on CPU (requires ~60 GB RAM for 27B)
python classify.py --device cpu
```

---

## Command-Line Options Reference

| Option | Values | Default | Description |
|---|---|---|---|
| `--image` | File path | `20250221_222154.jpg` | Path to target image |
| `--model` | `Cloudflare/clef`, `Cloudflare/clef-flash` | `Cloudflare/clef` | Model to load |
| `--device` | `cuda`, `auto`, `cpu` | Auto-detected | Device execution target |

---

## Output Format

The script evaluates two question types in a single forward pass:
1. **Categorical Selection (`choice`)**: Ranks all candidate categories (`human`, `cat`, `bike`, `car`) and outputs normalized probabilities and the winning prediction.
2. **Calibrated Verification (`noul`)**: Provides independent boolean probability scores (0%–100%) for each category (`is_human`, `is_cat`, `is_bike`, `is_car`).
