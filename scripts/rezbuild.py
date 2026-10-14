"""Rebuild the two L96A1 archives with a replacement texture and model, and verify the result.

    python -I scripts/rezbuild.py <texture.dtx> <model.ltb> [out_dir]

Reads the original archives from the client data folder (never modifies them), writes new ones to out_dir, then
re-reads them with the independent reader tools/rez.py and checks: same item list, every unchanged item
byte-identical, the replaced item equal to the new bytes.
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/sugarpestana/Desktop/untitled folder/tools")
import rez  # noqa: E402

from ca_pipeline import rezpack  # noqa: E402

G = "/Users/sugarpestana/Desktop/untitled folder/CombatArms/Game/"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/Users/sugarpestana/Desktop/CA-Textures/outputs/rez/"
os.makedirs(OUT, exist_ok=True)
jobs = {
    "GUNS_T_PV_SR.rez": {"GUNS_T_PV_SR/PV_SR_L96A1_BLACKMAGNUM_HEXA": open(sys.argv[1], "rb").read()},
    "GUNS_M_PV_SR.rez": {"GUNS_M_PV_SR/PV_SR_L96A1_SH": open(sys.argv[2], "rb").read()},
}
for name, new in jobs.items():
    orig = open(G + name, "rb").read()
    out = rezpack.replace(orig, new)
    p = OUT + name
    open(p, "wb").write(out)
    f0, i0 = rez.parse(G + name)
    f1, i1 = rez.parse(p)
    n0 = {n: (pos, sz) for n, pos, sz, t in i0}
    n1 = {n: (pos, sz, t) for n, pos, sz, t in i1}
    assert set(n0) == set(n1), "item lists differ"
    bad = same = 0
    for n, (pos, sz) in n0.items():
        a = orig[pos:pos + sz]
        q = n1[n]
        b = out[q[0]:q[0] + q[1]]
        if n in new:
            assert b == new[n], "replaced item mismatch"
        elif a == b:
            same += 1
        else:
            bad += 1
    print(f"{name}: items {len(n0)}, unchanged items identical {same}, mismatching {bad}, "
          f"header text identical {out[:0xC3] == orig[:0xC3]}, size {len(orig)} -> {len(out)}")
    assert bad == 0
print("all verified")
