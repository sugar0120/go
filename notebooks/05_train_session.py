# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "pillow", "torch>=2.7", "spandrel", "huggingface_hub"]
# ///
"""Step 5: one molab training session (repeat until training finishes).

molab limits this design works around:
- a session lasts at most 12 hours, and GPU sandboxes have been reclaimed
  early, so checkpoints are pushed to a private HF repo every 10 minutes and
  training stops itself 45 minutes before the 12-hour mark;
- only ~1 GB of notebook files persists, so the dataset and checkpoints are
  pulled to local disk at the start of every session;
- the machine has 4 CPUs, so the dataset is pre-built (notebook 04), never
  degraded on the fly; data loading would otherwise starve the GPU.

Needs: an HF_TOKEN secret with write access, and traiNNer-redux cloned into
the session (cell below) with your HAT-L config in it.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    import marimo as mo

    ROOT = mo.notebook_dir().parent
    sys.path.insert(0, str(ROOT))
    from ca_pipeline import gpu
    from ca_pipeline.sync import CheckpointSync, HFRemote, session_deadline
    return (CheckpointSync, HFRemote, Path, gpu, mo, os, session_deadline, subprocess,
            sys, time)


@app.cell
def _(gpu, mo):
    info = gpu.describe()
    mo.stop(not info["name"], mo.callout("No GPU attached. Turn it on from the notebook specs button "
                                         "in the app header, then restart.", kind="danger"))
    mo.vstack([
        mo.md(f"**GPU:** {info['name']}, {info['vram_gib']:.0f} GiB, compute {info['capability']}, "
              f"bf16: {info['bf16']}"),
        mo.callout(mo.md("<br>".join(info["issues"])), kind="danger") if info["issues"]
        else mo.callout("PyTorch can use this GPU.", kind="success"),
    ])
    return


@app.cell
def _(mo, os):
    repo = mo.ui.text(value="", placeholder="your-hf-user/ca-rebuild-training",
                      label="Private HF dataset repo", full_width=True)
    work = mo.ui.text(value="/tmp/ca_train", label="Local scratch dir (not persisted)", full_width=True)
    run_name = mo.ui.text(value="hat-l-stage-a", label="Run name (matches `name:` in the config)")
    config = mo.ui.text(value="options/train/ca_hat_l.yml", label="Config path inside traiNNer-redux",
                        full_width=True)
    # Check this flag against the traiNNer-redux README; it resumes from the
    # newest training state in experiments/<run name>/.
    command = mo.ui.text(value="python train.py --auto_resume -opt {config}", label="Train command",
                         full_width=True)
    start = mo.ui.run_button(label="Pull data + checkpoints, then train")
    has_token = bool(os.environ.get("HF_TOKEN"))
    mo.vstack([repo, work, run_name, config, command, start,
               mo.md("" if has_token else "**HF_TOKEN is not set**; add it as a secret first.")])
    return command, config, repo, run_name, start, work


@app.cell
def _(CheckpointSync, HFRemote, Path, command, config, mo, os, repo, run_name,
      session_deadline, start, subprocess, sys, time, work):
    mo.stop(not start.value)
    t_start = time.time()
    deadline = session_deadline(t_start, max_hours=12, margin_minutes=45)
    scratch = Path(work.value)
    trainer = scratch / "traiNNer-redux"
    remote = HFRemote(repo.value, token=os.environ.get("HF_TOKEN"))

    if not trainer.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/the-database/traiNNer-redux", str(trainer)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                        str(trainer / "requirements.txt")], check=True)
    remote.download("dataset", scratch / "dataset")
    remote.download("configs", trainer / "options" / "train")
    sync = CheckpointSync(remote, trainer / "experiments" / run_name.value,
                          prefix=f"runs/{run_name.value}", interval=600)
    sync.pull()
    sync.start()

    proc = subprocess.Popen(command.value.format(config=config.value), shell=True, cwd=trainer,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail: list[str] = []
    for line in proc.stdout:
        tail = (tail + [line.rstrip()])[-25:]
        mo.output.replace(mo.md("```\n" + "\n".join(tail) + "\n```"))
        if time.time() > deadline:
            proc.terminate()  # leave time for the final upload before molab's 12 h cutoff
            break
    proc.wait()
    pushed = sync.stop()
    mo.md(f"Session finished after {(time.time() - t_start) / 3600:.1f} h, exit code {proc.returncode}. "
          f"Final sync pushed {len(pushed)} file(s). "
          + (f"Last sync error: `{sync.last_error}`" if sync.last_error else "")
          + " Start a new session and run this again to continue.")
    return


if __name__ == "__main__":
    app.run()
