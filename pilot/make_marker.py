"""Make a red-tinted 1024 copy of a DTX: if the gun shows red in game, loose files override the .rez."""
import sys
import numpy as np
sys.path.insert(0, '.')
from ca_pipeline import dtx
src, dst = sys.argv[1:3]
d = dtx.parse(open(src, 'rb').read())
a = dtx.decode(d).astype(np.float32)
a[..., 0] = np.clip(a[..., 0] * 1.8 + 60, 0, 255)
a[..., 1] *= 0.4
a[..., 2] *= 0.4
open(dst, 'wb').write(dtx.write(d, a.astype(np.uint8)))
print('wrote', dst)
