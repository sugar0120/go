# Combat Arms BR: texture upgrade plan (the "70%" plan)

**Goal:** a clearly better-looking Combat Arms 2017 BR (private, not
distributed, still on LithTech Jupiter EX) for a small fraction of the work.
We aim for about 70% of the best possible result and skip the parts that cost
the most time for the least visible gain.

## What we do and what we skip

| Do | Skip (for now) |
|---|---|
| Upscale **textures** with 4x-UltraSharp, **fine-tuned briefly** on game-like textures (1–2 molab sessions) | The big multi-day HAT-L training run |
| **4x for first-person weapons**, **2x for everything else** | 4x on everything (four times the memory for little visible gain at a distance) |
| Fix alpha, tiling seams and normal maps automatically | Generating new PBR maps (LithTech doesn't use them) |
| Keep every **3D model** as it is | Remodeling, retopology, new LODs |
| Leave lightmaps and skies alone | Re-baking lighting |

Most of what players notice is blurry textures, above all on the gun that's
on screen the whole time. Upscaling textures fixes that without touching
meshes, rigs or animations, which is where the long work is.

## The model

1. **Base: 4x-UltraSharp** (ESRGAN, ~17M parameters). It's already good on
   game textures and trains fast. Download it from OpenModelDB into `models/`.
2. **Short fine-tune** on textures degraded the way Combat Arms textures
   were made (downscaled, then DXT-compressed). This teaches it to clean up
   *our* block artifacts instead of sharpening them. It's one stage, starts
   from the pretrained weights, and fits in 1–2 molab sessions (step 4).
3. **Keep it only if it wins:** in notebook 02, compare the fine-tune with
   stock UltraSharp, Real-ESRGAN and pretrained HAT-L on the same ~15 pilot
   textures. If the fine-tune doesn't clearly win, use the stock model. You
   lose nothing, because the batch step doesn't depend on training.

Everything runs comfortably on molab's GPU (RTX Pro 6000 Blackwell, 96 GB).
Why UltraSharp rather than HAT-L: it's 2–3x smaller and much faster to train
and run. HAT-L is the "more than 70%" option, listed under Later.

## Steps

### 1. Extract (1–2 days, on your PC)
- Unpack the `.rez` archives. Convert `.dtx` to PNG with a DTX converter or
  Noesis.
- Put the PNGs in `assets/converted/`, keeping the original folder structure.
- Keep an untouched copy of the original `.rez` files.

### 2. Inventory (minutes)
- Run `notebooks/01_inventory.py`. It writes `assets/manifest.csv` sorted by
  priority: weapons, characters, UI, props, world.

### 3. Pilot (half a day)
- Pick ~15 textures: 5 weapons, 3 characters, 3 walls/floors, 2 UI, 2 with
  transparency (fences, foliage).
- In `notebooks/02_benchmark_upscalers.py`, compare stock UltraSharp,
  Real-ESRGAN, HAT-L and Lanczos. This gives the baseline the fine-tune
  has to beat.
- Convert the 15 back to DTX, repack them, and look at them **in game**.

### 4. Short fine-tune (2–3 days, 1–2 molab sessions)
- **Data (1 day):** download **500–1,000** high-res CC0 textures (Poly Haven,
  ambientCG) in Combat Arms materials: metal, painted metal, gun parts,
  concrete, brick, wood, camo fabric, sand, foliage. Albedo maps only.
- **Build pairs (minutes):** `notebooks/04_build_training_set.py` with
  16 crops per image gives **~8k–16k pairs** (256 px targets, 64 px inputs).
  Upload them to the private HF repo from the same notebook.
- **Train:** `notebooks/05_train_session.py` with traiNNer-redux's ESRGAN
  template, starting from the 4x-UltraSharp weights:

  | Setting | Value |
  |---|---|
  | stages | **one** (the base is already GAN-trained) |
  | iterations | **~30k–40k** |
  | losses | L1 1.0, perceptual 1.0, GAN 0.1 (UNet discriminator) |
  | learning rate | 5e-5 (low, since it's a fine-tune), halved at 20k |
  | batch / patch | 32, 64 px input |
  | precision | bf16, EMA on |
  | checkpoint | every 1k iterations; validate every 2.5k |

  Expect roughly **6–15 hours** of GPU time. That's an estimate, so time the
  first 1k iterations. Notebook 05 resumes if molab cuts the session.
- **Pick (half a day):** load the 3 best validation checkpoints in notebook
  02, compare them on the pilot set against stock UltraSharp, and keep the
  winner.

### 5. Batch upscale on molab (1–3 sessions)
- Upload `assets/converted/` to a private Hugging Face repo, under `converted/`.
  molab keeps only about 1 GB of files between sessions.
- Run `notebooks/03_upscale_batch.py` one category at a time, in priority
  order. It pulls inputs from the repo, pushes results every 200 textures and
  skips textures that are already done. If molab ends the session early, just
  run it again.
- Use the model chosen in step 4 (the fine-tune or stock UltraSharp).
- Look through the QA table it prints (power-of-two, max size, lost alpha,
  broken normal maps) and the memory estimate.

### 6. Repack and playtest (1–2 days)
- Convert `assets/output/` back to DTX: **DXT1** for opaque textures and
  cutouts, **DXT5** for smooth transparency (`qa.pick_format`).
- Rebuild the REZ archives and play a few maps with several weapons.
- If something looks wrong, revert that texture to the original. Don't
  hand-fix it unless it's a hero asset.

**Total: about 2–3 weeks part-time.** Training adds 2–3 days, mostly
collecting textures; the GPU part runs unattended. Batch upscaling should
take hours of GPU time. These are estimates: time the pilot and the first 1k
training iterations, then multiply.

## Done when
- All weapon, character, UI, prop and world textures are upscaled and pass QA.
- The game loads every map without missing textures or crashes.
- Frame rate and load times are about the same as before on your normal PC.
- The game looks clearly sharper in side-by-side screenshots of the same spots.

## Risks to watch
- **LithTech limits:** the 2048 max size and power-of-two rule are
  assumptions. Test one 2048 weapon texture in the pilot. If it breaks, set
  `max_texture_size` in `ca_pipeline/config.py` to 1024.
- **Texture memory:** watch the memory estimate from step 5. If maps
  stutter, drop props and world textures back to the originals.
- **molab sessions:** they last up to 12 hours, but GPU sessions have been
  reported ending after 15–45 minutes. The batch notebook is resumable, so
  this only costs a restart.
- **Style drift:** AI models can make worn metal look too clean or invent
  text on decals. The pilot is there to catch this before the batch run.
- **Over-trained fine-tune:** if later checkpoints look plastic or
  oversharpened, use an earlier one. That's why one is saved every 1k
  iterations.

## Later, if you want more than 70%
These are already set up in the repo, but you don't need them for this plan:
- Full HAT-L fine-tune: a bigger model, two stages and 3–6 molab sessions
  (`docs/TRAINING.md`).
- 4x for characters (change `CATEGORY_POLICY` in `ca_pipeline/config.py`).
- Rework the 5–10 most-used weapon models in Blender, keeping bones and pivots.
