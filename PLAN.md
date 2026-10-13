# Combat Arms BR: texture upgrade plan (the "70%" plan)

**Goal:** a clearly better-looking Combat Arms 2017 BR (private, not
distributed, still on LithTech Jupiter EX) for a small fraction of the work.
We aim for about 70% of the best possible result and skip the parts that cost
the most time for the least visible gain.

## What we do and what we skip

| Do | Skip (for now) |
|---|---|
| Upscale **textures** with an off-the-shelf AI model | Training or fine-tuning our own model (days of GPU time) |
| **4x for first-person weapons**, **2x for everything else** | 4x on everything (four times the memory for little visible gain at a distance) |
| Fix alpha, tiling seams and normal maps automatically | Generating new PBR maps (LithTech doesn't use them) |
| Keep every **3D model** as it is | Remodeling, retopology, new LODs |
| Leave lightmaps and skies alone | Re-baking lighting |

Most of what players notice is blurry textures, above all on the gun that's
on screen the whole time. Upscaling textures fixes that without touching
meshes, rigs or animations, which is where the long work is.

## The model

1. **Default: 4x-UltraSharp** (ESRGAN). It's good on game textures, needs no
   training and is fast. Download it from OpenModelDB into `models/`.
2. **Quick check against 2 alternatives:** Real-ESRGAN x4plus and pretrained
   HAT-L (`Real_HAT_GAN_SRx4`). Run them on the same ~15 textures in notebook
   02 and keep whichever looks best. This takes an afternoon, not a project.
3. If none looks right on a category, use plain 2x Lanczos for that category
   and move on.

All three run comfortably on molab's GPU (RTX Pro 6000 Blackwell, 96 GB).

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
- In `notebooks/02_benchmark_upscalers.py`, compare UltraSharp, Real-ESRGAN,
  HAT-L and Lanczos, and pick one.
- Convert the 15 back to DTX, repack them, and look at them **in game**. Only
  continue once they look right there.

### 4. Batch upscale on molab (1–3 sessions)
- Upload `assets/converted/` to a private Hugging Face repo, under `converted/`.
  molab keeps only about 1 GB of files between sessions.
- Run `notebooks/03_upscale_batch.py` one category at a time, in priority
  order. It pulls inputs from the repo, pushes results every 200 textures and
  skips textures that are already done. If molab ends the session early, just
  run it again.
- Look through the QA table it prints (power-of-two, max size, lost alpha,
  broken normal maps) and the memory estimate.

### 5. Repack and playtest (1–2 days)
- Convert `assets/output/` back to DTX: **DXT1** for opaque textures and
  cutouts, **DXT5** for smooth transparency (`qa.pick_format`).
- Rebuild the REZ archives and play a few maps with several weapons.
- If something looks wrong, revert that texture to the original. Don't
  hand-fix it unless it's a hero asset.

**Total: about 1–2 weeks part-time**, most of it extraction and repacking.
The upscaling itself should take hours of GPU time. That's an estimate:
time the pilot and multiply by the texture count in the manifest.

## Done when
- All weapon, character, UI, prop and world textures are upscaled and pass QA.
- The game loads every map without missing textures or crashes.
- Frame rate and load times are about the same as before on your normal PC.
- The game looks clearly sharper in side-by-side screenshots of the same spots.

## Risks to watch
- **LithTech limits:** the 2048 max size and power-of-two rule are
  assumptions. Test one 2048 weapon texture in the pilot. If it breaks, set
  `max_texture_size` in `ca_pipeline/config.py` to 1024.
- **Texture memory:** watch the memory estimate from step 4. If maps
  stutter, drop props and world textures back to the originals.
- **molab sessions:** they last up to 12 hours, but GPU sessions have been
  reported ending after 15–45 minutes. The batch notebook is resumable, so
  this only costs a restart.
- **Style drift:** AI models can make worn metal look too clean or invent
  text on decals. The pilot is there to catch this before the batch run.

## Later, if you want more than 70%
These are already set up in the repo, but you don't need them for this plan:
- Fine-tune HAT-L on game-like textures (`notebooks/04`, `notebooks/05`,
  `docs/TRAINING.md`). It takes days of GPU time.
- 4x for characters (change `CATEGORY_POLICY` in `ca_pipeline/config.py`).
- Rework the 5–10 most-used weapon models in Blender, keeping bones and pivots.
