"""Keep copying a molab training run to the Mac so a reclaimed sandbox loses at most a few minutes.

    MOLAB_URL=.. MOLAB_TOKEN=.. python -I scripts/pull_loop.py <local_dir> [interval_seconds=180]

Each round (one short kernel call, training itself is detached): pull progress.txt, the newest ema_*_fp16.pth,
best.pth and any pilot outputs not yet copied. Stops when the DONE marker has been pulled or the box answers 410/refuses.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
local = sys.argv[1]
every = int(sys.argv[2]) if len(sys.argv) > 2 else 180
env = os.environ
os.makedirs(local, exist_ok=True)
LIST = """
import glob, os
c = sorted(glob.glob('/tmp/ca/run1/ema_*_fp16.pth'), key=lambda f: int(f.split('_')[-2]))
print('LATEST', c[-1] if c else 'none')
print('DONE', os.path.exists('/tmp/ca/run1/DONE'))
print('BEST', os.path.exists('/tmp/ca/run1/best.pth'))
"""
listing = os.path.join(local, "_list.py")
open(listing, "w").write(LIST)
log = open(os.path.join(local, "pull_loop.log"), "a")


def say(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


fails = 0
while True:
    r = subprocess.run([os.path.join(HERE, "mrun.sh"), listing], capture_output=True, text=True, env=env)
    out = r.stdout
    if "LATEST" not in out:
        fails += 1
        say("no answer from the box", fails, out.strip()[:120])
        if fails >= 5:
            say("giving up: box looks gone. Local copies are in", local)
            break
        time.sleep(30)
        continue
    fails = 0
    latest = [l.split()[1] for l in out.splitlines() if l.startswith("LATEST")][0]
    done = "DONE True" in out
    args = ["/tmp/ca/run1/progress.txt", os.path.join(local, "progress.txt")]
    if latest != "none":
        args += [latest, os.path.join(local, os.path.basename(latest))]
    if "BEST True" in out:
        args += ["/tmp/ca/run1/best.pth", os.path.join(local, "best.pth")]
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "vpull.py"), *args], capture_output=True, text=True, env=env)
    say("pulled", latest, "|", " ".join(l.split("/")[-1] for l in r.stdout.splitlines() if l.startswith("OK")))
    if done:
        say("DONE marker seen; final files pulled")
        break
    time.sleep(every)
