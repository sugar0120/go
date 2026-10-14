"""Side-by-side crops: original vs the gloss look.   python -I scripts/gloss_preview.py"""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
from PIL import Image
from ca_pipeline import dtx, texstyle
O = '/Users/sugarpestana/Desktop/CA-Textures/outputs/'
o = dtx.parse(open(O + 'PV_SR_DF_L96A1_AWSM_BLACK_BC.original.dtx', 'rb').read())
src = Image.fromarray(dtx.decode(o)[..., :3])
a = src.resize((2048, 2048), Image.LANCZOS)
b = texstyle.gloss(src, 2)
b.save('/Users/sugarpestana/.claude/jobs/27a6aafb/tmp/gloss_full.png')
crops = [(0, 100, 700), (1000, 0, 700), (300, 1300, 700), (0, 520, 700)]
W = 520
sheet = Image.new('RGB', (W * 2, W * len(crops)))
for r, (x, y, s) in enumerate(crops):
    for k, im in enumerate((a, b)):
        sheet.paste(im.crop((x, y, x + s, y + s)).resize((W, W), Image.LANCZOS), (k * W, r * W))
sheet.save('/Users/sugarpestana/.claude/jobs/27a6aafb/tmp/gloss_cmp.png'); print('ok')
