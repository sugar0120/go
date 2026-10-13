import sys,numpy as np
sys.path.insert(0,'.')
from ca_pipeline import dtx
from PIL import Image
src=sys.argv[1]
raw=open(src,'rb').read()
d=dtx.parse(raw)
print(d.format,d.width,d.height,'mips',d.mips,'iflags',d.iflags,'cmd',repr(d.command),'trailer',len(d.trailer))
a=dtx.decode(d)
ref=np.asarray(Image.open(sys.argv[2]).convert('RGBA'))
print('etcpak decode vs Pillow decode: max diff',int(np.abs(a.astype(int)-ref.astype(int)).max()))
out=dtx.write(d,a)
print('re-encode same size: bytes',len(out),'orig',len(raw),'header equal',out[:164]==raw[:164])
b=dtx.decode(dtx.parse(out))
mse=((a.astype(float)-b.astype(float))**2).mean();print('PSNR re-encode %.1f dB'%(10*np.log10(255**2/max(mse,1e-9))))
