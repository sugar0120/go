# Upscaling model: decision and training plan

These are image super-resolution models, not LLMs. They take a small texture
and output a larger one, and they're trained on pairs of low-res and high-res
images, not on text.

## Decision

| Role | Model | Why |
|---|---|---|
| **Production (hero + bulk)** | **HAT-L, fine-tuned on our own degradation**, starting from the `Real_HAT_GAN_SRx4` weights | Largest model loadable by `spandrel` (~41M params), so it has the highest detail ceiling. A big GPU makes its speed acceptable. Fine-tuning removes its main weakness: it was trained on photos, not on DXT-compressed game textures. |
| **Interim / fallback** | 4x-UltraSharp (ESRGAN) as-is | Good on game textures today with no training. Use it for the pilot while HAT-L trains; it's also the bar the fine-tune must beat. |
| Rejected | SUPIR (SDXL diffusion) | Biggest of all, but it invents detail and drifts from the original art. Results are inconsistent across thousands of textures, and it's impractical to fine-tune. At most, try it on a few hand-reviewed hero textures. |
| Rejected | Training from scratch | Needs far more data and compute than fine-tuning, for no gain. |

Confirm the choice with `notebooks/02_benchmark_upscalers.py`. The
fine-tuned HAT-L only becomes production once it beats 4x-UltraSharp on the
pilot set, both side by side and in the game.

## What training needs

### 1. Data (`notebooks/04_build_training_set.py`)
- **HR sources: 1,500–5,000 high-res images** in the materials Combat Arms
  uses: metal, painted and worn metal, gun parts, concrete, brick, wood,
  fabric and camo, desert ground, foliage. CC0 libraries (Poly Haven,
  ambientCG) are safe to use. Take the albedo/diffuse maps only.
- **Style anchors:** add the largest original game textures that are
  uncompressed (BGRA32 DTX). Don't use DXT-compressed originals as HR targets,
  or the model learns to reproduce the block artifacts.
- The notebook crops these into **256 px HR / 64 px LR pairs**, with 16 crops
  per source image, which gives roughly 25k–80k pairs. Every 20th source
  image goes to validation.
- LR inputs come from `ca_pipeline.degrade`: a random box, bilinear or bicubic
  downscale followed by a DXT1/DXT5 block-compression round-trip, matching how
  the original assets were made.

### 2. Trainer
Use **traiNNer-redux** (github.com/the-database/traiNNer-redux). It supports
HAT, ESRGAN, DAT and others, and saves weights that `spandrel`, and therefore
this pipeline, loads directly. Start from its bundled HAT-L training template
and set:

| Setting | Value |
|---|---|
| scale | 4 |
| dataset type | paired (`training/dataset/train/{hr,lr}`), val on `.../val/{hr,lr}` |
| LR patch size | 64 (HR 256) |
| batch size | 8 at 24 GB, 16+ at 48–80 GB |
| precision | bf16 AMP, with EMA weights |
| pretrained generator | `Real_HAT_GAN_SRx4` (strict load) |
| **Stage A** | ~50k iterations, L1 + perceptual loss, lr 1e-4 |
| **Stage B** | ~50–100k iterations, adds GAN loss (weight ~0.1) with a UNet discriminator, lr 1e-4 then halved at milestones |
| checkpoints | every 5k iterations, to persistent storage |
| validation | every 5k iterations: PSNR/SSIM plus LPIPS |

### 3. Hardware and time
- HAT-L training needs **at least 24 GB of VRAM**; 40–80 GB is comfortable.
- On one A100/H100-class GPU, expect roughly **1–3 days** for both stages.
  This is an estimate, so time the first 1k iterations and extrapolate.
- molab is a good fit for building the dataset, benchmarking and batch
  upscaling. Before running a multi-day training job there, check its session
  and runtime limits. If sessions can't run unattended that long, train on a
  rented GPU and bring the `.safetensors` back into `./models/`.

### 4. Picking the checkpoint
1. Shortlist 3–5 checkpoints with the best LPIPS on the validation set. LPIPS
   tracks what looks right better than PSNR does.
2. Load the shortlist in notebook 02 alongside 4x-UltraSharp and compare on
   the pilot textures.
3. Upscale the pilot set with the winner, repack it, and check it in game
   under real lighting.

### 5. Normal maps and alpha
- **Normal maps:** use the same model. `process_texture` renormalizes the
  vectors afterwards. Train a dedicated normal-map model only if QA shows
  artifacts.
- **Alpha:** upscaled with the same model as a separate channel. Cutouts are
  re-thresholded to hard edges.
