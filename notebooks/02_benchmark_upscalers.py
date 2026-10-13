# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "pillow", "torch>=2.7", "spandrel"]
# ///
"""Step 2: pick the upscaling model by measuring, not guessing.

Put candidate weights (.pth / .safetensors, see ca_pipeline/catalog.py) in
./models/, choose a handful of pilot textures, and compare quality, speed and
peak VRAM on this machine's GPU.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    from PIL import Image

    ROOT = mo.notebook_dir().parent
    sys.path.insert(0, str(ROOT))
    from ca_pipeline import catalog
    from ca_pipeline.textures import LanczosUpscaler, SpandrelUpscaler, process_texture
    return (Image, LanczosUpscaler, Path, ROOT, SpandrelUpscaler, catalog, mo,
            process_texture, time)


@app.cell
def _(catalog, mo):
    from ca_pipeline import gpu

    info = gpu.describe()
    try:
        import torch
    except ImportError:
        torch = None
    gpu_text = (f"{info['name']}, {info['vram_gib']:.0f} GiB" if info["name"]
                else "no CUDA GPU (CPU only); turn it on from the notebook specs button")
    mo.vstack([
        mo.md(f"**GPU:** {gpu_text}"),
        mo.callout(mo.md("<br>".join(info["issues"])), kind="danger") if info["issues"] else mo.md(""),
        mo.md("**Practical candidates here:** " + ", ".join(catalog.recommend_for_vram(info["vram_gib"]))),
        mo.ui.table([c.__dict__ for c in catalog.CANDIDATES], selection=None),
    ])
    return (torch,)


@app.cell
def _(Path, ROOT, mo):
    model_files = sorted(p for p in (ROOT / "models").glob("*") if p.suffix in (".pth", ".safetensors", ".pt"))
    pilot_files = sorted((ROOT / "assets" / "converted").rglob("*.png"))[:200]
    models_pick = mo.ui.multiselect([p.name for p in model_files], label="Models")
    pilot_pick = mo.ui.multiselect([str(p.relative_to(ROOT / "assets" / "converted")) for p in pilot_files],
                                   label="Pilot textures (pick 5-20 across categories)")
    run = mo.ui.run_button(label="Run benchmark")
    mo.vstack([models_pick, pilot_pick, run])
    return models_pick, pilot_pick, run


@app.cell
def _(Image, LanczosUpscaler, ROOT, SpandrelUpscaler, mo, models_pick, pilot_pick,
      process_texture, run, time, torch):
    mo.stop(not run.value, mo.md("Select models and textures, then press **Run benchmark**."))
    upscalers = [LanczosUpscaler(4)] + [SpandrelUpscaler(str(ROOT / "models" / m)) for m in models_pick.value]
    sources = {p: Image.open(ROOT / "assets" / "converted" / p) for p in pilot_pick.value}
    results, outputs = [], {}
    for up in upscalers:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _p, img in sources.items():
            outputs[(up.name, _p)] = process_texture(img, up)
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch is not None and torch.cuda.is_available() else 0
        results.append({"model": up.name, "scale": up.scale, "sec_per_texture": round(dt / max(len(sources), 1), 2),
                        "peak_vram_gib": round(peak, 2)})
    mo.ui.table(results, selection=None)
    return outputs, sources, upscalers


@app.cell
def _(mo, outputs, sources, upscalers):
    # Same texture, every model, at the same display size. Judge on the hero
    # assets you care about most; then confirm the winner in-engine.
    rows = []
    for _p in sources:
        rows.append(mo.md(f"#### {_p}"))
        rows.append(mo.hstack([mo.vstack([mo.image(outputs[(u.name, _p)], width=384), mo.md(u.name)])
                               for u in upscalers], wrap=True))
    mo.vstack(rows)
    return


if __name__ == "__main__":
    app.run()
