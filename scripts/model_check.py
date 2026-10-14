"""Round-trip and improve every supported first-person model of a weapon, with checks.   python -I scripts/model_check.py L96"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, "/Users/sugarpestana/Desktop/untitled folder/tools")
import rez  # noqa: E402

from ca_pipeline import ltbmesh, meshimprove  # noqa: E402

G = "/Users/sugarpestana/Desktop/untitled folder/CombatArms/Game/GUNS_M_PV_SR.rez"
f, items = rez.parse(G)
for n, pos, sz, t in items:
    base = n.split("/")[-1]
    if sys.argv[1].lower() not in base.lower() or not base.upper().startswith("PV_"):
        continue
    f.seek(pos)
    raw = f.read(sz)
    try:
        model = ltbmesh.parse_model(raw)
    except Exception as e:
        print(f"{base}: cannot parse ({str(e)[:70]})")
        continue
    rt = ltbmesh.serialize_model(model)
    stats = meshimprove.improve_model(model)
    probs = [meshimprove.check(m) if hasattr(m, "bonesets") else [] for m in model.meshes]
    out = ltbmesh.serialize_model(model)
    again = ltbmesh.parse_model(out)
    print(f"{base}: pieces {model.names} round trip identical {rt == raw}; "
          f"{[s['before'] for s in stats]} -> {[s['after'] for s in stats]}; reparse ok {len(again.meshes) == len(model.meshes)}; "
          f"suffix untouched {again.segments[-1] == model.segments[-1]}; problems {probs}")
