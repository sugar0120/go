"""Install / remove a loose DTX override in the CrossOver bottle (never edits a .rez).

    python -I pilot/install_loose.py install <file.dtx> <ARCHIVE> <NAME>   e.g. GUNS_T_PV_SR PV_SR_L96A1_BLACKMAGNUM_HEXA
    python -I pilot/install_loose.py undo <ARCHIVE> <NAME>
    python -I pilot/install_loose.py list

Target: <bottle>/Game/<ARCHIVE>/<NAME>.dtx  (whether the engine prefers it over the archive is what the marker test shows).
"""
import os
import shutil
import sys

GAME = os.path.expanduser('~/CXPBottles/CombatArms/drive_c/CombatArms/Game')
cmd = sys.argv[1]
if cmd == 'list':
    for d, _, fs in os.walk(GAME):
        for f in fs:
            if f.lower().endswith('.dtx') and os.path.relpath(d, GAME).upper().startswith(('GUNS_', 'CHARS_', 'TEXTURES', 'ATTACH', 'PROPS')):
                print(os.path.join(d, f))
elif cmd == 'install':
    src, arc, name = sys.argv[2:5]
    dst = os.path.join(GAME, arc, name + '.dtx')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    print('installed', dst, os.path.getsize(dst), 'bytes')
elif cmd == 'undo':
    arc, name = sys.argv[2:4]
    dst = os.path.join(GAME, arc, name + '.dtx')
    if os.path.exists(dst):
        os.remove(dst); print('removed', dst)
    d = os.path.dirname(dst)
    if os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d)
