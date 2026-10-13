# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "pillow", "torch", "spandrel"]
# ///
"""Step 3: run the chosen model over a category and QA the results.

Reads assets/manifest.csv (from step 1), writes PNGs to assets/output/ with
the same relative paths. Convert them back to DTX and repack REZ afterwards.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    import sys

    import marimo as mo
    from PIL import Image

    ROOT = mo.notebook_dir().parent
    sys.path.insert(0, str(ROOT))
    from ca_pipeline import qa
    from ca_pipeline.config import CATEGORY_POLICY
    from ca_pipeline.textures import LanczosUpscaler, SpandrelUpscaler, process_texture
    return (CATEGORY_POLICY, Image, LanczosUpscaler, ROOT, SpandrelUpscaler, csv, mo,
            process_texture, qa)


@app.cell
def _(CATEGORY_POLICY, ROOT, csv, mo):
    with open(ROOT / "assets" / "manifest.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r["kind"] == "texture" and r["width"]]
    model_names = ["lanczos"] + sorted(p.name for p in (ROOT / "models").glob("*")
                                       if p.suffix in (".pth", ".safetensors", ".pt"))
    category = mo.ui.dropdown(sorted(CATEGORY_POLICY, key=lambda c: CATEGORY_POLICY[c].priority),
                              value="weapon", label="Category")
    model = mo.ui.dropdown(model_names, value=model_names[-1], label="Model (winner from step 2)")
    tiling = mo.ui.checkbox(label="Treat world textures as tiling", value=True)
    run = mo.ui.run_button(label="Upscale category")
    mo.vstack([category, model, tiling, run])
    return category, model, rows, run, tiling


@app.cell
def _(Image, LanczosUpscaler, ROOT, SpandrelUpscaler, category, mo, model,
      process_texture, qa, rows, run, tiling):
    mo.stop(not run.value)
    up = (LanczosUpscaler(4) if model.value == "lanczos"
          else SpandrelUpscaler(str(ROOT / "models" / model.value)))
    # Scale-1 categories (lightmaps, skies) are re-baked or replaced, not upscaled.
    todo = [r for r in rows if r["category"] == category.value and int(r["target_scale"]) > 1]
    issues, total_bytes = [], 0
    for r in mo.status.progress_bar(todo, title=f"Upscaling {category.value}"):
        src_path = ROOT / "assets" / "converted" / r["path"]
        dst_path = (ROOT / "assets" / "output" / r["path"]).with_suffix(".png")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        src = Image.open(src_path)
        is_tiling = tiling.value and r["category"] == "world"
        out = process_texture(src, up, role=r["role"], tiling=is_tiling,
                              target_scale=int(r["target_scale"]))
        out.save(dst_path)
        issues += qa.check_texture(src, out, r["path"], role=r["role"])
        fmt = qa.pick_format(out.mode == "RGBA")
        total_bytes += qa.texture_bytes(*out.size, fmt)
    mo.vstack([
        mo.md(f"**{len(todo)} textures** written. Estimated DXT memory: "
              f"**{total_bytes / 2**20:.0f} MiB**."),
        mo.ui.table([i.__dict__ for i in issues], selection=None) if issues
        else mo.callout("All QA checks passed.", kind="success"),
    ])
    return


if __name__ == "__main__":
    app.run()
