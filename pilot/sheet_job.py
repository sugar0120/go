"""Runs on molab: build a crop comparison sheet (original nearest-4x | lanczos | model variants) and print it as base64."""
import base64
import io
import sys

from PIL import Image

sys.path.insert(0, '/tmp/ca/code')
from ca_pipeline import dtx

out = '/tmp/ca/out'
variants = [v for v in sys.argv[1:]] if len(sys.argv) > 1 else ['lanczos2x', 'ultrasharp4x']
orig = Image.fromarray(dtx.decode(dtx.parse(open('/tmp/ca/in/original.dtx', 'rb').read()))[..., :3])
S = 4096
base = orig.resize((S, S), Image.NEAREST)
# crops in 4096 space (x, y, size): scope lens, receiver camo + screws, engraved text
crops = [(150, 380, 512), (2000, 120, 512), (1200, 700, 512)]
cols = [('orig x4 nearest', base)] + [(v, Image.open(f'{out}/{v}.png').resize((S, S), Image.NEAREST if '2x' in v else Image.LANCZOS)) for v in variants]
W = 400
sheet = Image.new('RGB', (W * len(cols), W * len(crops)))
for r, (x, y, s) in enumerate(crops):
    for c, (name, im) in enumerate(cols):
        sheet.paste(im.crop((x, y, x + s, y + s)).resize((W, W), Image.LANCZOS), (c * W, r * W))
buf = io.BytesIO()
sheet.save(buf, 'JPEG', quality=88)
print('B64START' + base64.b64encode(buf.getvalue()).decode() + 'B64END')
print('cols', [n for n, _ in cols])
