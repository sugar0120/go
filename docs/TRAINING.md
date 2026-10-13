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
| batch size | 32 on molab's 96 GB GPU. Lower it if the 4 CPUs can't keep the GPU busy (GPU utilization under ~80%) |
| precision | bf16 AMP (native on Blackwell), with EMA weights |
| pretrained generator | `Real_HAT_GAN_SRx4` (strict load) |
| **Stage A** | ~50k iterations, L1 + perceptual loss, lr 1e-4 |
| **Stage B** | ~50–100k iterations, adds GAN loss (weight ~0.1) with a UNet discriminator, lr 1e-4 then halved at milestones |
| checkpoints | every 1k iterations (molab GPU sessions can end early). Notebook 05 uploads them |
| validation | every 5k iterations: PSNR/SSIM plus LPIPS |

### 3. Running on molab

**What molab provides** (public preview; marimo says these may change):

| | |
|---|---|
| GPU | NVIDIA RTX Pro 6000 **Blackwell**, 96 GB GDDR7. Turn it on from the notebook specs button |
| CPU / RAM | 4 CPUs, 32 GB |
| Session | 12 hours max. Shut down after 90 minutes idle |
| Persistent files | about 1 GB of notebook files. Anything larger is lost at shutdown |
| Cost | free while in preview |

**How this pipeline works within those limits:**
- **Blackwell:** needs PyTorch 2.7+ built for CUDA 12.8+. Notebooks 02 and 05
  check this at startup.
- **12-hour sessions:** notebook 05 runs one session, stops training 45
  minutes before the 12-hour cutoff, and uploads the last checkpoint. The next
  session resumes from it. Plan on **3–6 sessions** for both stages, roughly
  1–3 days of GPU time (an estimate, so time the first 1k iterations).
- **Early GPU reclaims:** an open marimo issue (marimo-team/marimo#11058,
  October 2026) reports GPU sandboxes reclaimed after 15–45 minutes. So
  checkpoints are saved every 1k iterations and pushed every 10 minutes, and
  the most you can lose is about 10 minutes of training.
- **1 GB persistence:** the dataset (several GB) and checkpoints (around
  0.5 GB each with optimizer state) live in a **private Hugging Face dataset
  repo**. Notebook 04 uploads the dataset, and notebook 05 pulls the dataset
  and the latest checkpoint into `/tmp` at the start of each session. Upload
  your trainer config to `configs/` in that repo. Store an `HF_TOKEN` with
  write access as a molab secret.
- **4 CPUs:** never degrade images on the fly during training. The dataset is
  pre-built, and the trainer only loads PNGs. If GPU utilization is low,
  lower the batch size or the dataloader worker count rather than adding
  augmentation.
- **90-minute idle shutdown:** it's unclear whether a running cell counts as
  activity. Keep the notebook open, and check on your first run that it
  survives past 90 minutes.

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
