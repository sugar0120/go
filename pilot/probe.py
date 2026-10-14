import os
import shutil
import subprocess
import sys
import time
import urllib.request

print('cpus', os.cpu_count(), 'py', sys.version.split()[0])
for p in ('/tmp', '/home', '/'):
    try:
        u = shutil.disk_usage(p)
        print('disk', p, 'free GB', round(u.free / 1e9), 'total', round(u.total / 1e9))
    except Exception as e:
        print(p, e)
print(subprocess.run('nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader; free -g | head -2',
                     shell=True, capture_output=True, text=True).stdout)
try:
    import torch
    print('torch', torch.__version__, torch.cuda.is_available())
except Exception as e:
    print('torch', e)
for u in ('https://huggingface.co', 'https://ambientcg.com', 'https://api.polyhaven.com/assets?t=textures&c=metal',
          'https://github.com'):
    t = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'x'}), timeout=20)
        print(u, r.status, round(time.time() - t, 2))
    except Exception as e:
        print(u, 'ERR', str(e)[:60])
print('HF_TOKEN set:', bool(os.environ.get('HF_TOKEN')))
