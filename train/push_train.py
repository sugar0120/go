"""Print Python that, run on the molab kernel, stages the training scripts and starts a detached job.

    python -I train/push_train.py dataset  > train/_p.py   # then: scripts/mrun.sh train/_p.py
    python -I train/push_train.py train [extra finetune args...]
"""
import base64
import gzip
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mode, extra = sys.argv[1], " ".join(sys.argv[2:])
files = {"train/build_dataset.py": "train/build_dataset.py", "train/finetune.py": "train/finetune.py",
         "code/ca_pipeline/dtx.py": "ca_pipeline/dtx.py"}
print("import base64, gzip, os, subprocess, sys")
print("os.makedirs('/tmp/ca/train', exist_ok=True); os.makedirs('/tmp/ca/code/ca_pipeline', exist_ok=True)")
for dst, src in files.items():
    b = base64.b64encode(gzip.compress(open(os.path.join(root, src), "rb").read(), 9)).decode()
    print(f"open('/tmp/ca/{dst}','wb').write(gzip.decompress(base64.b64decode('{b}')))")
print("subprocess.run([sys.executable,'-m','pip','install','-q','lpips','spandrel','etcpak'],capture_output=True)")
if mode == "dataset":
    print(f"cmd = 'nohup ' + sys.executable + ' /tmp/ca/train/build_dataset.py /tmp/ca/code /tmp/ca/ds {extra} > /tmp/ca/ds.log 2>&1 &'")
else:
    print("cmd = 'nohup ' + sys.executable + ' /tmp/ca/train/finetune.py --data /tmp/ca/ds --model /tmp/ca/models/4x-UltraSharp.pth "
          f"--out /tmp/ca/run1 {extra} > /tmp/ca/train.log 2>&1 &'")
print("subprocess.Popen(cmd, shell=True, start_new_session=True)")
print("print('started:', cmd)")
