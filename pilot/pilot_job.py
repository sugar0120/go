"""Pilot job for ONE Combat Arms texture. Runs on molab (GPU) or locally with --no-ai.

    python pilot_job.py <code_dir> <in.dtx> <out_dir> [--no-ai] [--model path.pth]

Writes, next to a report.json:
  lanczos2x.dtx       plain resampling, 2048 (baseline, also the engine-limit test file)
  ultrasharp2x.dtx    4x-UltraSharp, downscaled to 2048
  ultrasharp4x.dtx    4x-UltraSharp at full 4096
plus a PNG of each. Every .dtx keeps the original header, flags and command
string (see ca_pipeline/dtx.py).
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image

code_dir, src, out = sys.argv[1:4]
no_ai = "--no-ai" in sys.argv
model_path = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None
sys.path.insert(0, code_dir)
from ca_pipeline import dtx  # noqa: E402

os.makedirs(out, exist_ok=True)
orig = dtx.parse(open(src, "rb").read())
rgba = dtx.decode(orig)
h, w = rgba.shape[:2]
opaque = bool((rgba[..., 3] == 255).all())
report = {"source": os.path.basename(src), "format": orig.format, "size": [w, h], "opaque": opaque, "variants": {}}
rgb = rgba[..., :3].astype(np.float32) / 255.0


def finish(name, rgb_out):
    a = (np.clip(rgb_out, 0, 1) * 255 + 0.5).astype(np.uint8)
    a = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
    t = time.time()
    data = dtx.write(orig, a)
    Image.fromarray(a[..., :3]).save(os.path.join(out, name + ".png"))
    open(os.path.join(out, name + ".dtx"), "wb").write(data)
    report["variants"][name] = {"size": [a.shape[1], a.shape[0]], "dtx_bytes": len(data),
                                "encode_s": round(time.time() - t, 2)}


def resize(x, size):
    return np.stack([np.asarray(Image.fromarray(x[..., c], mode="F").resize(size, Image.LANCZOS))
                     for c in range(3)], -1)


finish("lanczos2x", resize(rgb, (w * 2, h * 2)))

if not no_ai:
    import torch
    from ca_pipeline.textures import SpandrelUpscaler

    report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    up = SpandrelUpscaler(model_path, tile=512, overlap=32)
    t = time.time()
    big = up(rgb)  # 4x
    report["inference_s"] = round(time.time() - t, 2)
    finish("ultrasharp4x", big)
    finish("ultrasharp2x", resize(big, (w * 2, h * 2)))

json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=1)
print(json.dumps(report))
