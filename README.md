# Combat Arms (2017 BR) asset rebuild

A pipeline for raising the quality of textures and models in a **private,
non-distributed** rebuild of Combat Arms 2017 BR. The engine stays **LithTech
Jupiter EX**, so every output has to fit its limits: power-of-two textures, a
maximum size (2048 by default), DXT compression, and polygon counts close to
the originals. Engine limits and per-category upscale scales live in
`ca_pipeline/config.py`. Verify them during the pilot.

The work runs in **marimo notebooks**, which open in molab or run locally.

## Workflow

| Step | Where | What |
|---|---|---|
| 0. Extract | outside this repo | Unpack the `.rez` archives with a REZ extractor. Convert `.dtx` to PNG/DDS and `.ltb` to OBJ/FBX (Noesis with LithTech plugins, or community DTX/LTB converters). Originals go in `assets/original/` (read-only), converted files in `assets/converted/`. |
| 1. Inventory | `notebooks/01_inventory.py` | Scans `assets/converted/`, guesses category and role (diffuse/normal/spec), and writes `assets/manifest.csv` in priority order: weapons, characters, UI, props, world. |
| 2. Choose model | `notebooks/02_benchmark_upscalers.py` | Compares candidate models (`ca_pipeline/catalog.py`) on pilot textures: side by side, seconds per texture, peak VRAM. |
| 3. Upscale + QA | `notebooks/03_upscale_batch.py` | Upscales one category at a time into `assets/output/`. Handles alpha, tiling seams and normal maps, fits to engine limits, and runs QA checks and a VRAM estimate. |
| 4. Fine-tune | `notebooks/04_build_training_set.py` + `docs/TRAINING.md` | Builds HR/LR training pairs that mimic the game's DXT textures, for fine-tuning the production model. |
| 5. Repack | outside this repo | Convert PNG back to DTX (DXT1 for opaque or cutout, DXT5 for smooth alpha, see `qa.pick_format`), rebuild the REZ archives, and test in the client. |

**Model decision:** a fine-tuned HAT-L for production, with 4x-UltraSharp
as the interim model and the bar to beat. The reasoning and the full training
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

## Layout

```
ca_pipeline/   config (engine limits), manifest, textures, degrade, qa, catalog
notebooks/     01-04 marimo notebooks
docs/          TRAINING.md
assets/        original/ converted/ work/ output/  (tracked with Git LFS)
tests/         pytest suite
```

Run `git lfs install` before adding game assets. `.gitattributes` routes
`.rez/.dtx/.ltb/.dds/.png/.tga/.fbx/.obj` under `assets/` through LFS.
