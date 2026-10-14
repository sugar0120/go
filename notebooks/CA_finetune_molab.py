# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo"]
# ///
"""Combat Arms texture fine-tune, run interactively on molab.

Open from GitHub on molab, attach the GPU (notebook specs button), then run the cells top to bottom.
Everything runs in cells with the editor open (live progress, run buttons, download buttons), not as a
detached job. Checkpoints go to a private Hugging Face repo when you give a token, so a reclaimed
sandbox resumes from the last one instead of starting over.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Combat Arms texture fine-tune (molab)

        1. Attach the GPU (notebook specs button), then run **Setup**.
        2. Optional: paste a Hugging Face **write** token and a private repo name so checkpoints survive a reclaim.
        3. **Build dataset**, then **Train**. If the sandbox is reclaimed, open a new one and run the same cells:
           it resumes from the newest checkpoint in the Hugging Face repo.
        4. **Try it** on a texture, then use the download buttons.

        The token is kept in memory only. Do not paste secrets into code cells: this notebook is public by link.
        """
    )
    return (mo,)


@app.cell
def _(mo):
    branch = mo.ui.text(value="worktree-l96-pilot", label="Repo branch (sugar0120/go)")
    hf_repo = mo.ui.text(value="", placeholder="your-hf-user/ca-textures", label="Private HF dataset repo (optional)")
    hf_token = mo.ui.text(value="", kind="password", label="HF write token (optional)")
    iters = mo.ui.slider(500, 12000, step=500, value=4000, label="Training iterations")
    ckpt_every = mo.ui.slider(100, 1000, step=100, value=200, label="Checkpoint every N steps")
    setup_btn = mo.ui.run_button(label="Setup")
    mo.vstack([branch, hf_repo, hf_token, iters, ckpt_every, setup_btn])
    return branch, ckpt_every, hf_repo, hf_token, iters, setup_btn


@app.cell
def _(branch, mo, setup_btn):
    mo.stop(not setup_btn.value, mo.md("Click **Setup**."))
    import os
    import subprocess
    import sys
    import urllib.request

    ROOT = "/tmp/ca/go"
    os.makedirs("/tmp/ca/models", exist_ok=True)
    if not os.path.exists(ROOT):
        subprocess.run(["git", "clone", "--depth", "1", "-b", branch.value, "https://github.com/sugar0120/go", ROOT],
                       check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "spandrel", "lpips", "etcpak", "huggingface_hub"],
                   capture_output=True)
    MODEL = "/tmp/ca/models/4x-UltraSharp.pth"
    if not os.path.exists(MODEL):
        urllib.request.urlretrieve("https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth", MODEL)
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    mo.callout(mo.md(f"GPU: **{gpu}**" if gpu else "**No GPU attached.** Use the notebook specs button, then rerun."),
               kind="success" if gpu else "danger")
    return MODEL, ROOT, os, subprocess, sys


@app.cell
def _(mo):
    ds_assets = mo.ui.slider(40, 150, step=10, value=70, label="CC0 materials per source (ambientCG, Poly Haven)")
    ds_btn = mo.ui.run_button(label="Build dataset")
    mo.vstack([ds_assets, ds_btn])
    return ds_assets, ds_btn


@app.cell
def _(ROOT, ds_assets, ds_btn, mo, os, subprocess, sys):
    mo.stop(not ds_btn.value, mo.md("Click **Build dataset** (a few minutes)."))
    _cmd = [sys.executable, f"{ROOT}/train/build_dataset.py", ROOT, "/tmp/ca/ds", "--ambient", str(ds_assets.value),
            "--polyhaven", str(ds_assets.value), "--crops", "24"]
    _p = subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    _tail = []
    for _line in _p.stdout:
        _tail = (_tail + [_line.rstrip()])[-8:]
        mo.output.replace(mo.md("```\n" + "\n".join(_tail) + "\n```"))
    _p.wait()
    _n = len(os.listdir("/tmp/ca/ds/train/hr")) if os.path.isdir("/tmp/ca/ds/train/hr") else 0
    mo.md(f"Dataset ready: **{_n}** training pairs.")
    return


@app.cell
def _(mo):
    train_btn = mo.ui.run_button(label="Train (resumes from the HF repo if it has a checkpoint)")
    train_btn
    return (train_btn,)


@app.cell
def _(MODEL, ROOT, ckpt_every, hf_repo, hf_token, iters, mo, os, subprocess, sys, train_btn):
    mo.stop(not train_btn.value, mo.md("Click **Train**."))
    _env = dict(os.environ)
    _model, _start = MODEL, 0
    if hf_token.value and hf_repo.value:
        _env["HF_TOKEN"] = hf_token.value
        try:
            from huggingface_hub import HfApi, hf_hub_download

            _api = HfApi(token=hf_token.value)
            _names = [f for f in _api.list_repo_files(hf_repo.value, repo_type="dataset")
                      if f.startswith("run1/ema_") and f.endswith("_fp16.pth")]
            if _names:
                _best = max(_names, key=lambda f: int(f.split("_")[-2]))
                _model = hf_hub_download(hf_repo.value, _best, repo_type="dataset", token=hf_token.value,
                                         local_dir="/tmp/ca/resume")
                _start = int(_best.split("_")[-2])
        except Exception as _e:  # no repo yet is fine
            print("resume check:", str(_e)[:100])
    _cmd = [sys.executable, f"{ROOT}/train/finetune.py", "--data", "/tmp/ca/ds", "--model", _model, "--out",
            "/tmp/ca/run1", "--iters", str(iters.value), "--save-every", str(ckpt_every.value), "--val-every",
            str(ckpt_every.value), "--start-step", str(_start)]
    if hf_repo.value:
        _cmd += ["--hf-repo", hf_repo.value]
    mo.output.replace(mo.md(f"Starting from step **{_start}**."))
    _p = subprocess.Popen(_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_env)
    _tail = []
    for _line in _p.stdout:
        if "Warning" in _line or "warn" in _line or "%|" in _line:
            continue
        _tail = (_tail + [_line.rstrip()])[-14:]
        mo.output.replace(mo.md(f"Starting from step **{_start}**.\n```\n" + "\n".join(_tail) + "\n```"))
    _p.wait()
    mo.md(f"Training process exited with code {_p.returncode}.")
    return


@app.cell
def _(mo):
    upload = mo.ui.file(filetypes=[".dtx"], label="Upload the original .dtx to try the model on")
    scale = mo.ui.dropdown({"2x (2048)": 2, "4x (4096)": 4}, value="2x (2048)", label="Output scale")
    try_btn = mo.ui.run_button(label="Upscale with the fine-tuned model (and the stock model for comparison)")
    mo.vstack([upload, scale, try_btn])
    return scale, try_btn, upload


@app.cell
def _(MODEL, ROOT, mo, os, scale, sys, try_btn, upload):
    mo.stop(not (try_btn.value and upload.value), mo.md("Upload a .dtx and click the button."))
    import glob
    import io

    import numpy as np
    from PIL import Image

    sys.path.insert(0, ROOT)
    from ca_pipeline import dtx
    from ca_pipeline.textures import SpandrelUpscaler

    orig_dtx = dtx.parse(upload.contents())
    _rgba = dtx.decode(orig_dtx)
    _rgb = _rgba[..., :3].astype(np.float32) / 255.0
    _h, _w = _rgb.shape[:2]
    _s = scale.value

    def _run(weights):
        _up = SpandrelUpscaler(weights, tile=512, overlap=32)
        _big = _up(_rgb)
        if _s != 4:
            _big = np.stack([np.asarray(Image.fromarray(_big[..., c], mode="F").resize((_w * _s, _h * _s), Image.LANCZOS))
                             for c in range(3)], -1)
        _a = (np.clip(_big, 0, 1) * 255 + 0.5).astype(np.uint8)
        return np.concatenate([_a, np.full(_a.shape[:2] + (1,), 255, np.uint8)], -1)

    _ft = sorted(glob.glob("/tmp/ca/run1/ema_*.pth"), key=lambda f: int(f.split("_")[-1].split(".")[0].replace("fp16", "") or 0))
    _best = "/tmp/ca/run1/best.pth" if os.path.exists("/tmp/ca/run1/best.pth") else None
    stock_out = _run(MODEL)
    tuned_out = _run(_best) if _best else None
    _crop = (_w * _s // 4, _h * _s // 4, 256)

    def _view(a):
        return Image.fromarray(a[..., :3]).crop((_crop[0], _crop[1], _crop[0] + _crop[2], _crop[1] + _crop[2])).resize((384, 384), Image.NEAREST)

    mo.hstack([mo.vstack([mo.md("stock"), _view(stock_out)]),
               mo.vstack([mo.md("fine-tuned" if _best else "fine-tuned: not trained yet"),
                          _view(tuned_out) if tuned_out is not None else mo.md("")])])
    return dtx, orig_dtx, stock_out, tuned_out


@app.cell
def _(dtx, mo, orig_dtx, os, stock_out, tuned_out):
    # DTX files keep the original header, flags and command string (ca_pipeline/dtx.py).
    _items = [mo.download(data=dtx.write(orig_dtx, stock_out), filename="stock_ultrasharp.dtx", label="stock .dtx")]
    if tuned_out is not None:
        _items.append(mo.download(data=dtx.write(orig_dtx, tuned_out), filename="finetuned.dtx", label="fine-tuned .dtx"))
    if os.path.exists("/tmp/ca/run1/best.pth"):
        _items.append(mo.download(data=open("/tmp/ca/run1/best.pth", "rb").read(), filename="best.pth",
                                  label="fine-tuned weights"))
    mo.hstack(_items, justify="start")
    return


if __name__ == "__main__":
    app.run()
