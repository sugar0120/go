# Combat Arms (2017 BR) asset rebuild

A pipeline for raising the quality of textures and models in a **private,
non-distributed** rebuild of Combat Arms 2017 BR. The engine stays **LithTech
Jupiter EX**, so every output has to fit its limits: power-of-two textures, a
maximum size (2048 by default), DXT compression, and polygon counts close to
the originals. Engine limits and per-category upscale scales live in
`ca_pipeline/config.py`. Verify them during the pilot.

The work runs in **marimo notebooks**, which open in molab or run locally.

**Start with [`PLAN.md`](PLAN.md)**, the lighter "70%" plan: pretrained models,
textures only, about 1–2 weeks part-time. The fine-tuning steps (4–5) below are
optional extras.

## Workflow

| Step | Where | What |
|---|---|---|
| 0. Extract | outside this repo | Unpack the `.rez` archives with a REZ extractor. Convert `.dtx` to PNG/DDS and `.ltb` to OBJ/FBX (Noesis with LithTech plugins, or community DTX/LTB converters). Originals go in `assets/original/` (read-only), converted files in `assets/converted/`. |
| 1. Inventory | `notebooks/01_inventory.py` | Scans `assets/converted/`, guesses category and role (diffuse/normal/spec), and writes `assets/manifest.csv` in priority order: weapons, characters, UI, props, world. |
| 2. Choose model | `notebooks/02_benchmark_upscalers.py` | Compares candidate models (`ca_pipeline/catalog.py`) on pilot textures: side by side, seconds per texture, peak VRAM. |
| 3. Upscale + QA | `notebooks/03_upscale_batch.py` | Upscales one category at a time into `assets/output/`. Handles alpha, tiling seams and normal maps, fits to engine limits, and runs QA checks and a VRAM estimate. |
| 4. Fine-tune | `notebooks/04_build_training_set.py` + `docs/TRAINING.md` | Builds HR/LR training pairs that mimic the game's DXT textures, for fine-tuning the production model. |
| 5. Train (molab) | `notebooks/05_train_session.py` | One 12-hour molab session of training on the Blackwell GPU. Pulls the dataset and checkpoints from a private HF repo, pushes checkpoints every 10 minutes, and stops before the cutoff. Rerun it until training finishes. |
| 6. Repack | outside this repo | Convert PNG back to DTX (DXT1 for opaque or cutout, DXT5 for smooth alpha, see `qa.pick_format`), rebuild the REZ archives, and test in the client. |

**Model decision:** 4x-UltraSharp, pretrained (see `PLAN.md`). A fine-tuned
HAT-L is the optional upgrade if you want more. The reasoning and the full training
recipe are in [`docs/TRAINING.md`](docs/TRAINING.md).

**Models (meshes):** export to OBJ and rework in Blender, keeping bone
names, hierarchy and pivots identical so the existing animations still work.
Check triangle budgets with `qa.obj_stats` / `qa.check_mesh`. The default
allows 2x the original triangle count.

## Running

Locally:

```bash
pip install -e ".[dev]"          # core + tests
pip install -e ".[ai]"           # torch + spandrel, for AI upscalers
marimo edit notebooks/01_inventory.py
pytest
```

On molab, open a notebook from this GitHub repo. Each notebook declares its
own dependencies in its header and imports `ca_pipeline` from the repo root.
Model weights go in `./models/`, which is git-ignored.

molab gives each notebook an RTX Pro 6000 Blackwell (96 GB, turn it on from
the notebook specs button), 4 CPUs and 32 GB of RAM, for up to 12 hours per
session. Only about 1 GB of files persists, so datasets and checkpoints live in a
private Hugging Face repo. See [`docs/TRAINING.md`](docs/TRAINING.md#3-running-on-molab).

## Layout

```
ca_pipeline/   config (engine limits), manifest, textures, degrade, qa, catalog, gpu, sync
notebooks/     01-05 marimo notebooks
docs/          TRAINING.md
assets/        original/ converted/ work/ output/  (tracked with Git LFS)
tests/         pytest suite
```

Run `git lfs install` before adding game assets. `.gitattributes` routes
`.rez/.dtx/.ltb/.dds/.png/.tga/.fbx/.obj` under `assets/` through LFS.
