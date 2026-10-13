# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "pillow"]
# ///
"""Step 4 (fine-tuning): build HR/LR pairs that look like Combat Arms textures.

HR sources: high-res CC0 textures (Poly Haven, ambientCG) in matching
materials, plus the largest original game textures. LR inputs are made with
ca_pipeline.degrade (downscale + DXT block compression). See docs/TRAINING.md.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    import marimo as mo
    import numpy as np
    from PIL import Image

    ROOT = mo.notebook_dir().parent
    sys.path.insert(0, str(ROOT))
    from ca_pipeline.degrade import degrade, random_crops
    return Image, ROOT, degrade, mo, np, random_crops


@app.cell
def _(ROOT, mo):
    src = mo.ui.text(value=str(ROOT / "training" / "hr_sources"), label="HR source folder", full_width=True)
    dst = mo.ui.text(value=str(ROOT / "training" / "dataset"), label="Output folder", full_width=True)
    crop = mo.ui.slider(128, 512, step=64, value=256, label="HR crop size")
    per_image = mo.ui.slider(1, 64, value=16, label="Crops per source image")
    val_every = mo.ui.slider(10, 100, value=20, label="Every Nth source goes to validation")
    run = mo.ui.run_button(label="Build dataset")
    mo.vstack([src, dst, crop, per_image, val_every, run])
    return crop, dst, per_image, run, src, val_every


@app.cell
def _(Image, crop, degrade, dst, mo, np, per_image, random_crops, run, src, val_every):
    mo.stop(not run.value)
    from pathlib import Path

    rng = np.random.default_rng(1234)  # fixed seed: reproducible dataset
    files = sorted(p for p in Path(src.value).rglob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tga", ".dds"))
    counts = {"train": 0, "val": 0}
    for i, f in enumerate(mo.status.progress_bar(files, title="Cropping and degrading")):
        split = "val" if i % val_every.value == 0 else "train"
        img = Image.open(f).convert("RGB")
        for j, hr in enumerate(random_crops(img, crop.value, per_image.value, rng)):
            name = f"{f.stem}_{j:03d}.png"
            for sub, im in (("hr", hr), ("lr", degrade(hr, scale=4, rng=rng))):
                out = Path(dst.value) / split / sub / name
                out.parent.mkdir(parents=True, exist_ok=True)
                im.save(out)
            counts[split] += 1
    mo.md(f"Built **{counts['train']}** training and **{counts['val']}** validation pairs "
          f"from {len(files)} source images.")
    return


if __name__ == "__main__":
    app.run()
