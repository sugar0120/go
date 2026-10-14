"""Build HR/LR training pairs from CC0 textures (ambientCG + Poly Haven).

    python build_dataset.py <code_dir> <out_dir> [--ambient N] [--polyhaven N] [--crops 16] [--seed 1]

LR = HR crop downscaled 4x (random BOX/BILINEAR/BICUBIC) and then run through the REAL DXT1 encoder
and decoder (etcpak + Pillow), so the model sees the same block artifacts the game's textures have.
Only albedo/colour maps are used. Both sources are CC0; the list of what was used goes to sources.json.
Images are processed and discarded as they download, so disk use stays small.
Output: <out_dir>/{train,val}/{hr,lr}/<name>.png   (HR 256 px, LR 64 px)
"""
import io
import json
import os
import random
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

code_dir, out_dir = sys.argv[1:3]
opt = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d  # noqa: E731
N_AMB, N_PH, CROPS, SEED = opt("--ambient", 120), opt("--polyhaven", 100), opt("--crops", 16), opt("--seed", 1)
sys.path.insert(0, code_dir)
from ca_pipeline import dtx  # noqa: E402

UA = {"User-Agent": "ca-texture-pipeline/0.1 (private research)"}
HR = 256
AMB_PREFIXES = ("Metal", "PaintedMetal", "Rust", "Concrete", "Bricks", "Wood", "Planks", "Fabric", "Ground",
                "Gravel", "Asphalt", "Plaster", "Tiles", "Rock", "Leather", "Concrete", "Marble", "Paint")
PH_CATEGORIES = ("metal", "concrete", "brick", "wood", "fabric", "rock", "sand", "planks", "floor", "wall")


def get(url, tries=3):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()
        except Exception:
            if i == tries - 1:
                raise


def dxt1_roundtrip(img):
    rgba = np.asarray(img.convert("RGBA"))
    h, w = rgba.shape[:2]
    raw = dtx._encode_pixels(rgba, dtx.BPP_DXT1)
    return Image.fromarray(dtx._decode_s3tc(raw, w, h, dtx.BPP_DXT1)[..., :3])


def make_pairs(img, name, rng, split):
    img = img.convert("RGB")
    if rng.random() < 0.5:  # vary detail density: game textures cover similar real-world area with fewer pixels
        s = rng.uniform(0.5, 1.0)
        img = img.resize((max(HR, int(img.width * s)), max(HR, int(img.height * s))), Image.LANCZOS)
    w, h = img.size
    if w < HR or h < HR:
        return 0
    n = 0
    for j in range(CROPS):
        x, y = rng.randrange(0, w - HR + 1), rng.randrange(0, h - HR + 1)
        hr = img.crop((x, y, x + HR, y + HR))
        k = rng.choice([Image.BOX, Image.BILINEAR, Image.BICUBIC])
        lr = dxt1_roundtrip(hr.resize((HR // 4, HR // 4), k))
        for sub, im in (("hr", hr), ("lr", lr)):
            p = os.path.join(out_dir, split, sub, f"{name}_{j:02d}.png")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            im.save(p)
        n += 1
    return n


def ambient_list(limit):
    out, offset = [], 0
    while True:
        d = json.loads(get(f"https://ambientcg.com/api/v2/full_json?type=Material&limit=100&offset={offset}"
                           "&include=downloadData"))
        for a in d["foundAssets"]:
            if a["assetId"].startswith(AMB_PREFIXES):
                try:
                    files = {f["attribute"]: f["downloadLink"] for f in
                             a["downloadFolders"]["default"]["downloadFiletypeCategories"]["zip"]["downloads"]}
                except (KeyError, TypeError):
                    continue
                url = files.get("2K-PNG") or files.get("2K-JPG")
                if url:
                    out.append((a["assetId"], url))
        offset += 100
        if offset >= d["numberOfResults"]:
            break
    random.Random(SEED).shuffle(out)
    return out[:limit]


def polyhaven_list(limit):
    seen = {}
    for c in PH_CATEGORIES:
        try:
            for k in json.loads(get(f"https://api.polyhaven.com/assets?t=textures&c={c}")):
                seen[k] = c
        except Exception as e:
            print("polyhaven category failed", c, e)
    ids = sorted(seen)
    random.Random(SEED).shuffle(ids)
    return ids[:limit]


def do_ambient(item):
    aid, url = item
    rng = random.Random(f"{SEED}{aid}")
    z = zipfile.ZipFile(io.BytesIO(get(url)))
    col = [n for n in z.namelist() if "_Color." in n]
    if not col:
        return aid, 0, None
    split = "val" if rng.random() < 0.05 else "train"
    return aid, make_pairs(Image.open(io.BytesIO(z.read(col[0]))), "amb_" + aid, rng, split), url


def do_polyhaven(pid):
    rng = random.Random(f"{SEED}{pid}")
    files = json.loads(get(f"https://api.polyhaven.com/files/{pid}"))
    d = files.get("Diffuse", {}).get("2k", {})
    url = (d.get("png") or d.get("jpg") or {}).get("url")
    if not url:
        return pid, 0, None
    split = "val" if rng.random() < 0.05 else "train"
    return pid, make_pairs(Image.open(io.BytesIO(get(url))), "ph_" + pid, rng, split), url


sources, total = [], 0
for label, items, fn in (("ambientCG", ambient_list(N_AMB) if N_AMB else [], do_ambient),
                         ("polyhaven", polyhaven_list(N_PH) if N_PH else [], do_polyhaven)):
    print(label, len(items), "assets", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for aid, n, url in ex.map(fn, items):
            total += n
            sources.append({"source": label, "id": aid, "pairs": n, "url": url})
            if len(sources) % 20 == 0:
                print(len(sources), "assets,", total, "pairs", flush=True)
json.dump(sources, open(os.path.join(out_dir, "sources.json"), "w"), indent=1)
print("DONE pairs", total, "assets", len(sources), flush=True)
