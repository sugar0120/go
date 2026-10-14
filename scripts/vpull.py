"""Copy files off a molab box over plain HTTP using marimo virtual-file URLs.

    MOLAB_URL=.. MOLAB_TOKEN=.. python -I scripts/vpull.py <remote_path> <local_path> [<remote_path> <local_path> ...]

One kernel call registers the files and prints their /@file/ URLs; curl downloads them at full speed (the old
base64-through-console route managed ~22 KB/s). The URLs are unguessable but NOT authenticated: use this for weights
trained on CC0 data and for outputs you are happy to hold briefly behind a random link. Run only when no other kernel
call is in flight (training itself runs detached, so it is not affected).
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
url, tok = os.environ["MOLAB_URL"].rstrip("/"), os.environ["MOLAB_TOKEN"]
pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
code = """
import os, re, marimo as mo
for p in %r:
    if not os.path.exists(p):
        print('MISSING', p); continue
    d = mo.download(data=open(p, 'rb').read(), filename=os.path.basename(p), mimetype='application/octet-stream')
    m = re.search(r'data-data=.&quot;(\\./@file/[^&]+)&quot;', d.text)
    print('VF', p, m.group(1)[2:] if m else 'NOURL', os.path.getsize(p))
""" % ([r for r, _ in pairs],)
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
    f.write(code)
out = subprocess.run([os.path.join(HERE, "mrun.sh"), f.name], capture_output=True, text=True,
                     env={**os.environ, "MOLAB_URL": url + "/", "MOLAB_TOKEN": tok}).stdout
os.unlink(f.name)
got = {}
for line in out.splitlines():
    if line.startswith("VF "):
        _, rp, path, size = line.split()
        got[rp] = (path, int(size))
    elif line.startswith("MISSING"):
        print(line)
ok = True
for rp, lp in pairs:
    if rp not in got or got[rp][0] == "NOURL":
        print("not available:", rp)
        ok = False
        continue
    os.makedirs(os.path.dirname(os.path.abspath(lp)), exist_ok=True)
    r = subprocess.run(["curl", "-sf", "-m", "600", "-o", lp, f"{url}/{got[rp][0]}"])
    size = os.path.getsize(lp) if os.path.exists(lp) else -1
    good = r.returncode == 0 and size == got[rp][1]
    print(("OK  " if good else "BAD ") + lp, size, "of", got[rp][1])
    ok = ok and good
sys.exit(0 if ok else 1)
