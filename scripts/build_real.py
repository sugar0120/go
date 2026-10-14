"""Build the real upgraded archives for one weapon skin: improved texture + improved model.

    python -I scripts/build_real.py <texture item> <mesh item> [--scale 2] [--tex-file upgraded.dtx] [out_dir]

e.g. PV_SR_DF_L96A1_AWSM_BLACK_BC  PV_SR_L96A1_SUPER_MAGNUM_SH
If --tex-file is not given, the texture is upscaled with plain Lanczos and then sharpened (no neural model): useful
for an immediate in-game test. Pass the fine-tuned model's output with --tex-file to use that instead.
Originals are read from the client data folder and never modified; results are verified with tools/rez.py.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/sugarpestana/Desktop/untitled folder/tools")
import rez  # noqa: E402

from ca_pipeline import dtx, ltbmesh, meshimprove, rezpack  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = {sys.argv[i]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
tex_name, mesh_name = args[0], args[1]
scale = int(opt.get("--scale", 2))
OUT = args[2] if len(args) > 2 else "/Users/sugarpestana/Desktop/CA-Textures/outputs/rez_real/"
G = "/Users/sugarpestana/Desktop/untitled folder/CombatArms/Game/"
os.makedirs(OUT, exist_ok=True)

# ---- texture
tsrc = open(G + "GUNS_T_PV_SR.rez", "rb").read()
troot, _ = rezpack.parse(tsrc)
tpath = [p for p, it in rezpack.walk(troot) if p.endswith("/" + tex_name)][0]
tit = dict(rezpack.walk(troot))[tpath]
orig = dtx.parse(tsrc[tit.pos:tit.pos + tit.size])
if "--tex-file" in opt:
    new_tex = open(opt["--tex-file"], "rb").read()
    print("texture: using", opt["--tex-file"], dtx.parse(new_tex).width)
elif opt.get("--style") == "gloss":
    from ca_pipeline import texstyle

    arr = np.asarray(texstyle.gloss(Image.fromarray(dtx.decode(orig)[..., :3]), scale))
    new_tex = dtx.write(orig, np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1))
    print("texture: gloss style,", orig.width, "->", orig.width * scale)
else:
    im = Image.fromarray(dtx.decode(orig)[..., :3]).resize((orig.width * scale, orig.height * scale), Image.LANCZOS)

    def luma(i, fn):
        y, cb, cr = i.convert("YCbCr").split()
        return Image.merge("YCbCr", (fn(y), cb, cr)).convert("RGB")

    k = scale / 2
    im = luma(im, lambda y: y.filter(ImageFilter.UnsharpMask(radius=1.4 * k, percent=120, threshold=1)))
    im = luma(im, lambda y: y.filter(ImageFilter.UnsharpMask(radius=14 * k, percent=45, threshold=2)))
    x = np.asarray(im).astype(np.float32) / 255
    x = np.clip(np.power(x, 0.85), 0, 1)               # gentle shadow lift: this skin is almost black
    arr = (x * 255 + 0.5).astype(np.uint8)
    new_tex = dtx.write(orig, np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1))
    print("texture: Lanczos + sharpen + shadow lift,", orig.width, "->", orig.width * scale)
open(OUT + tex_name + ".dtx", "wb").write(new_tex)

# ---- model
msrc = open(G + "GUNS_M_PV_SR.rez", "rb").read()
mroot, _ = rezpack.parse(msrc)
mpath = [p for p, it in rezpack.walk(mroot) if p.endswith("/" + mesh_name)][0]
mit = dict(rezpack.walk(mroot))[mpath]
model = ltbmesh.parse_model(msrc[mit.pos:mit.pos + mit.size])
assert ltbmesh.serialize_model(model) == msrc[mit.pos:mit.pos + mit.size], "round trip not identical"
stats = meshimprove.improve_model(model)
new_mesh = ltbmesh.serialize_model(model)
for m in model.meshes:
    assert int(m.tris.max()) < m.n_verts and np.isfinite(m.pos).all()
print("model:", [(s["before"], s["after"]) for s in stats])
open(OUT + mesh_name + ".ltb", "wb").write(new_mesh)

# ---- archives + independent verification
for name, orig_b, new in (("GUNS_T_PV_SR.rez", tsrc, {tpath: new_tex}), ("GUNS_M_PV_SR.rez", msrc, {mpath: new_mesh})):
    out = rezpack.replace(orig_b, new)
    open(OUT + name, "wb").write(out)
    f0, i0 = rez.parse(G + name)
    f1, i1 = rez.parse(OUT + name)
    n0 = {n: (p, s) for n, p, s, t in i0}
    n1 = {n: (p, s) for n, p, s, t in i1}
    assert set(n0) == set(n1)
    bad = sum(1 for n in n0 if n not in new and orig_b[n0[n][0]:n0[n][0] + n0[n][1]] != out[n1[n][0]:n1[n][0] + n1[n][1]])
    ok = all(out[n1[n][0]:n1[n][0] + n1[n][1]] == v for n, v in new.items())
    print(f"{name}: replaced {list(new)}  others identical: {bad == 0}  replaced read back: {ok}  size {len(orig_b)} -> {len(out)}")
    assert bad == 0 and ok
