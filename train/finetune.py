"""Fine-tune 4x-UltraSharp (ESRGAN/RRDB) on Combat Arms-style degraded texture pairs. Runs on a molab GPU.

    python finetune.py --data /tmp/ca/ds --model /tmp/ca/models/4x-UltraSharp.pth --out /tmp/ca/train \
        [--iters 6000] [--bs 32] [--lr 5e-5] [--gan 0.05] [--save-every 1000] [--val-every 1000]

Self-contained (no traiNNer-redux): the network is loaded with spandrel and trained directly.
Loss: L1 + perceptual (LPIPS-VGG) + optional GAN (UNet-SN discriminator, starts after --gan-start).
Validation uses a DIFFERENT metric (LPIPS-AlexNet + PSNR) on held-out pairs, and compares with the stock
weights, so "better" is measured against the model we started from, not against the training loss.
Resumable: re-run the same command and it continues from the newest checkpoint in --out.
Writes progress.txt (one line per log), ema_<step>.pth (loadable by spandrel), best.pth and a DONE marker.
"""
import argparse
import copy
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--iters", type=int, default=6000)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--gan", type=float, default=0.05)
p.add_argument("--gan-start", type=int, default=500)
p.add_argument("--perc", type=float, default=1.0)
p.add_argument("--ema", type=float, default=0.999)
p.add_argument("--save-every", type=int, default=1000)
p.add_argument("--val-every", type=int, default=1000)
p.add_argument("--max-val", type=int, default=256)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
logf = open(os.path.join(a.out, "progress.txt"), "a")


def log(*x):
    s = " ".join(str(i) for i in x)
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def load_split(split, limit=None):
    names = sorted(os.listdir(os.path.join(a.data, split, "hr")))
    if limit:
        names = names[:: max(1, len(names) // limit)][:limit]
    hr = np.stack([np.asarray(Image.open(os.path.join(a.data, split, "hr", n)).convert("RGB")) for n in names])
    lr = np.stack([np.asarray(Image.open(os.path.join(a.data, split, "lr", n)).convert("RGB")) for n in names])
    t = lambda x: torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()  # noqa: E731
    return t(lr).to(dev), t(hr).to(dev)


class UNetDiscriminatorSN(nn.Module):
    """Real-ESRGAN's U-Net discriminator with spectral norm."""

    def __init__(self, nf=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.c0 = nn.Conv2d(3, nf, 3, 1, 1)
        self.c1 = sn(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False))
        self.c2 = sn(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False))
        self.c3 = sn(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False))
        self.c4 = sn(nn.Conv2d(nf * 8, nf * 4, 3, 1, 1, bias=False))
        self.c5 = sn(nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=False))
        self.c6 = sn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=False))
        self.c7 = sn(nn.Conv2d(nf, nf, 3, 1, 1, bias=False))
        self.c8 = sn(nn.Conv2d(nf, nf, 3, 1, 1, bias=False))
        self.c9 = nn.Conv2d(nf, 1, 3, 1, 1)

    def forward(self, x):
        act = lambda t: F.leaky_relu(t, 0.2, inplace=True)  # noqa: E731
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)  # noqa: E731
        x0 = act(self.c0(x))
        x1 = act(self.c1(x0))
        x2 = act(self.c2(x1))
        x3 = act(self.c3(x2))
        x4 = act(self.c4(up(x3))) + x2
        x5 = act(self.c5(up(x4))) + x1
        x6 = act(self.c6(up(x5))) + x0
        return self.c9(act(self.c8(act(self.c7(x6)))))


def to_f(x):  # uint8 NCHW -> float [0,1]
    return x.float() / 255.0


def augment(lr, hr):
    if np.random.rand() < 0.5:
        lr, hr = lr.flip(3), hr.flip(3)
    if np.random.rand() < 0.5:
        lr, hr = lr.flip(2), hr.flip(2)
    k = np.random.randint(4)
    return torch.rot90(lr, k, (2, 3)), torch.rot90(hr, k, (2, 3))


from spandrel import ModelLoader  # noqa: E402
import lpips  # noqa: E402

desc = ModelLoader().load_from_file(a.model)
net = desc.model.to(dev)
assert desc.scale == 4, desc.scale
for q in net.parameters():
    q.requires_grad_(True)
stock = copy.deepcopy(net).eval()
for q in stock.parameters():
    q.requires_grad_(False)
ema = copy.deepcopy(net).eval()
for q in ema.parameters():
    q.requires_grad_(False)
D = UNetDiscriminatorSN().to(dev)
opt_g = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(0.9, 0.99))
opt_d = torch.optim.Adam(D.parameters(), lr=a.lr * 2, betas=(0.9, 0.99))
perc_loss = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
val_metric = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
for m in (perc_loss, val_metric):
    for q in m.parameters():
        q.requires_grad_(False)

tr_lr, tr_hr = load_split("train")
va_lr, va_hr = load_split("val", a.max_val)
log(f"data: train {len(tr_lr)} pairs, val {len(va_lr)} pairs, params {sum(q.numel() for q in net.parameters())/1e6:.1f}M")


@torch.no_grad()
def evaluate(model):
    ps, lp = [], []
    for i in range(0, len(va_lr), 16):
        x, y = to_f(va_lr[i:i + 16]), to_f(va_hr[i:i + 16])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sr = model(x)
        sr = sr.float().clamp(0, 1)
        mse = ((sr - y) ** 2).mean(dim=(1, 2, 3))
        ps.append(10 * torch.log10(1.0 / mse.clamp_min(1e-10)))
        lp.append(val_metric(sr * 2 - 1, y * 2 - 1).flatten())
    return torch.cat(ps).mean().item(), torch.cat(lp).mean().item()


def ckpts():
    return sorted(glob.glob(os.path.join(a.out, "ckpt_*.pt")), key=lambda f: int(f.split("_")[-1][:-3]))


step, best = 0, 1e9
cs = ckpts()
if cs:
    ck = torch.load(cs[-1], map_location=dev)
    net.load_state_dict(ck["net"])
    ema.load_state_dict(ck["ema"])
    D.load_state_dict(ck["D"])
    opt_g.load_state_dict(ck["opt_g"])
    opt_d.load_state_dict(ck["opt_d"])
    step, best = ck["step"], ck["best"]
    log(f"resumed from step {step}")
else:
    sp, sl = evaluate(stock)
    log(f"STOCK val: psnr {sp:.2f} lpips {sl:.4f}")

t0 = time.time()
run = {"l1": 0.0, "perc": 0.0, "g": 0.0, "d": 0.0, "n": 0}
bce = nn.BCEWithLogitsLoss()
while step < a.iters:
    net.train()
    if step >= a.iters // 2:
        for g in opt_g.param_groups:
            g["lr"] = a.lr * 0.5
    idx = torch.randint(0, len(tr_lr), (a.bs,), device=dev)
    lr_b, hr_b = augment(tr_lr[idx], tr_hr[idx])
    lr_b, hr_b = to_f(lr_b), to_f(hr_b)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sr = net(lr_b)
    sr = sr.float()
    l1 = F.l1_loss(sr, hr_b)
    lp = perc_loss(sr.clamp(0, 1) * 2 - 1, hr_b * 2 - 1).mean()
    loss = l1 + a.perc * lp
    use_gan = a.gan > 0 and step >= a.gan_start
    if use_gan:
        for q in D.parameters():
            q.requires_grad_(False)
        fake_g = D(sr)
        gl = bce(fake_g, torch.ones_like(fake_g))
        loss = loss + a.gan * gl
    opt_g.zero_grad(set_to_none=True)
    loss.backward()
    opt_g.step()
    if use_gan:
        for q in D.parameters():
            q.requires_grad_(True)
        real, fake = D(hr_b), D(sr.detach())
        dl = bce(real, torch.ones_like(real)) + bce(fake, torch.zeros_like(fake))
        opt_d.zero_grad(set_to_none=True)
        dl.backward()
        opt_d.step()
        run["g"] += gl.item()
        run["d"] += dl.item()
    with torch.no_grad():
        for pe, pn in zip(ema.parameters(), net.parameters()):
            pe.mul_(a.ema).add_(pn.detach(), alpha=1 - a.ema)
    step += 1
    run["l1"] += l1.item()
    run["perc"] += lp.item()
    run["n"] += 1
    if step % 100 == 0:
        n = run["n"]
        log(f"step {step} l1 {run['l1']/n:.4f} perc {run['perc']/n:.4f} g {run['g']/n:.3f} d {run['d']/n:.3f} "
            f"{step/(time.time()-t0+1e-9):.2f} it/s(session-avg)")
        run = {k: 0.0 for k in run}
        run["n"] = 0
    if step % a.val_every == 0 or step == a.iters:
        ep, el = evaluate(ema)
        log(f"VAL step {step}: psnr {ep:.2f} lpips {el:.4f}  (best lpips {min(best, el):.4f})")
        if el < best:
            best = el
            torch.save(ema.state_dict(), os.path.join(a.out, "best.pth"))
    if step % a.save_every == 0 or step == a.iters:
        torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(), "step": step, "best": best}, os.path.join(a.out, f"ckpt_{step}.pt"))
        torch.save({k: v.half() for k, v in ema.state_dict().items()}, os.path.join(a.out, f"ema_{step}_fp16.pth"))  # small enough to pull off the box
        torch.save(ema.state_dict(), os.path.join(a.out, f"ema_{step}.pth"))
        for old in ckpts()[:-2]:
            os.remove(old)
        log(f"saved step {step}")

# export check: the saved EMA weights must load back through spandrel and give the same output
sd = torch.load(os.path.join(a.out, f"ema_{a.iters}.pth"), map_location=dev)
re = ModelLoader().load_from_state_dict(sd).model.to(dev).eval()
with torch.no_grad():
    x = to_f(va_lr[:2])
    same = torch.allclose(re(x), ema(x), atol=1e-4)
log("export reload check:", "OK" if same else "MISMATCH")
open(os.path.join(a.out, "DONE"), "w").write("done")
log("DONE")
