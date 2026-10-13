"""Print Python that, run on the molab kernel, installs deps, stages the pilot and starts it detached.

    python -I pilot/push_pilot.py | scripts/mrun.sh -c "$(cat)"      # MOLAB_URL / MOLAB_TOKEN in env

No secrets are included. The model is downloaded on molab straight from Hugging Face (fast there).
Poll with:  scripts/mrun.sh -c "print(open('/tmp/ca/job.log').read()[-2000:])"
Results land in /tmp/ca/out (ephemeral: pull them before the session ends).
"""
import base64
import gzip
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = {
    "code/ca_pipeline/__init__.py": "ca_pipeline/__init__.py",
    "code/ca_pipeline/config.py": "ca_pipeline/config.py",
    "code/ca_pipeline/dtx.py": "ca_pipeline/dtx.py",
    "code/ca_pipeline/textures.py": "ca_pipeline/textures.py",
    "pilot_job.py": "pilot/pilot_job.py",
    "in/original.dtx": "pilot/original_PV_SR_L96A1_BLACKMAGNUM_HEXA.dtx",
}
print("import base64, gzip, os, subprocess, sys, shutil")
print("os.makedirs('/tmp/ca/models', exist_ok=True); os.makedirs('/tmp/ca/in', exist_ok=True); os.makedirs('/tmp/ca/code/ca_pipeline', exist_ok=True)")
for dst, src in files.items():
    b = base64.b64encode(gzip.compress(open(os.path.join(root, src), "rb").read(), 9)).decode()
    print(f"open('/tmp/ca/{dst}','wb').write(gzip.decompress(base64.b64decode('{b}')))")
print("""
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'spandrel', 'etcpak'], capture_output=True, text=True)
if r.returncode:
    uv = shutil.which('uv')
    r = subprocess.run([uv, 'pip', 'install', '-q', '--python', sys.executable, 'spandrel', 'etcpak'], capture_output=True, text=True) if uv else r
print('pip rc', r.returncode, r.stderr[-300:])
import urllib.request
m = '/tmp/ca/models/4x-UltraSharp.pth'
if not os.path.exists(m):
    urllib.request.urlretrieve('https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth', m)
print('model bytes', os.path.getsize(m))
subprocess.Popen(f'nohup {sys.executable} /tmp/ca/pilot_job.py /tmp/ca/code /tmp/ca/in/original.dtx /tmp/ca/out --model {m} > /tmp/ca/job.log 2>&1 &',
                 shell=True, start_new_session=True)
print('started')
""")
