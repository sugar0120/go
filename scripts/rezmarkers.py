"""Find which texture/mesh the game really uses for a weapon: tint EVERY matching texture a different colour and
scale EVERY matching first-person mesh, in rebuilt copies of the two archives (originals untouched).

    python -I scripts/rezmarkers.py L96 [out_dir]

Prints a legend (item -> colour). Whatever colour the owner sees on the gun in a match names the file in use.
"""
import colorsys
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/sugarpestana/Desktop/untitled folder/tools")
import rez  # noqa: E402

from ca_pipeline import dtx, ltbmesh, rezpack  # noqa: E402

KEY = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/Users/sugarpestana/Desktop/CA-Textures/outputs/rez_marker/"
G = "/Users/sugarpestana/Desktop/untitled folder/CombatArms/Game/"
os.makedirs(OUT, exist_ok=True)
PALETTE = [("red", 0.0), ("orange", 0.07), ("yellow", 0.15), ("lime", 0.25), ("green", 0.35), ("teal", 0.47),
           ("cyan", 0.52), ("blue", 0.62), ("purple", 0.75), ("magenta", 0.85), ("pink", 0.93), ("white", None),
           ("black", None), ("brown", 0.05), ("olive", 0.20), ("navy", 0.66), ("gold", 0.12), ("violet", 0.80),
           ("sky", 0.56), ("rose", 0.97)]


def solid(rgba, name, hue):
    """Replace the colours with a flat colour that keeps the original light/dark detail."""
    lum = rgba[..., :3].astype(float).mean(-1) / 255.0
    if name == "white":
        rgb = np.stack([np.clip(lum * 0.3 + 0.7, 0, 1)] * 3, -1)
    elif name == "black":
        rgb = np.stack([lum * 0.15] * 3, -1)
    else:
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        rgb = np.clip((0.25 + 0.75 * lum)[..., None] * np.array([r, g, b]), 0, 1)
    out = rgba.copy()
    out[..., :3] = (rgb * 255 + 0.5).astype(np.uint8)
    return out


# textures
src = open(G + "GUNS_T_PV_SR.rez", "rb").read()
root, _ = rezpack.parse(src)
tex = {p: it for p, it in rezpack.walk(root) if KEY.lower() in p.lower()}
new_t, legend = {}, []
for k, (path, it) in enumerate(sorted(tex.items())):
    d = dtx.parse(src[it.pos:it.pos + it.size])
    name, hue = PALETTE[k % len(PALETTE)]
    try:
        new_t[path] = dtx.write(d, solid(dtx.decode(d), name, hue))
        legend.append((name, path, f"{d.width}x{d.height} {d.format}"))
    except NotImplementedError as e:
        legend.append(("(skipped)", path, f"{d.width}x{d.height} {d.format}: {e}"))
open(OUT + "GUNS_T_PV_SR.rez", "wb").write(rezpack.replace(src, new_t))

# first-person meshes (PV_ files only; ANI_ files are skipped)
srcm = open(G + "GUNS_M_PV_SR.rez", "rb").read()
rootm, _ = rezpack.parse(srcm)
new_m, skipped = {}, []
for path, it in rezpack.walk(rootm):
    base = path.split("/")[-1]
    if KEY.lower() in base.lower() and base.upper().startswith("PV_"):
        try:
            f, _, _ = ltbmesh.parse(srcm[it.pos:it.pos + it.size])
        except Exception as e:  # unsupported variant: leave it alone
            skipped.append((path, str(e)[:60]))
            continue
        c = f.mesh.pos.mean(0)
        f.mesh.pos = ((f.mesh.pos - c) * 1.6 + c).astype(np.float32)
        new_m[path] = ltbmesh.serialize(f)
open(OUT + "GUNS_M_PV_SR.rez", "wb").write(rezpack.replace(srcm, new_m))

# verify both with the independent reader
for name, orig, new in (("GUNS_T_PV_SR.rez", src, new_t), ("GUNS_M_PV_SR.rez", srcm, new_m)):
    f0, i0 = rez.parse(G + name)
    f1, i1 = rez.parse(OUT + name)
    out = open(OUT + name, "rb").read()
    n0 = {n: (p, s) for n, p, s, t in i0}
    n1 = {n: (p, s) for n, p, s, t in i1}
    assert set(n0) == set(n1)
    bad = sum(1 for n in n0 if n not in new and orig[n0[n][0]:n0[n][0] + n0[n][1]] != out[n1[n][0]:n1[n][0] + n1[n][1]])
    ok = all(out[n1[n][0]:n1[n][0] + n1[n][1]] == v for n, v in new.items())
    print(f"{name}: replaced {len(new)}, unchanged-but-different {bad}, replaced read back correctly {ok}")
    assert bad == 0 and ok
print("\nTEXTURE COLOUR LEGEND")
for name, path, info in legend:
    print(f"  {name:8s} {path.split('/')[-1]:50s} {info}")
print("\nMESHES SCALED 1.6x:", [p.split('/')[-1] for p in new_m], "skipped:", skipped)
