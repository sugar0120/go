import struct

import numpy as np
import pytest

pytest.importorskip("etcpak")
from ca_pipeline import dtx  # noqa: E402


def _fake(w=64, h=64, bpp=dtx.BPP_DXT1, cmd=b"alpharef 128"):
    hdr = bytearray(164)
    struct.pack_into("<I", hdr, 0, 0)
    struct.pack_into("<HH", hdr, 4, w, h)
    struct.pack_into("<i", hdr, 8, -5)
    struct.pack_into("<HH", hdr, 12, 1, 0)
    struct.pack_into("<I", hdr, 16, 8 | (1 << 7))
    hdr[24], hdr[25], hdr[26] = 3, 1, bpp
    hdr[36:36 + len(cmd)] = cmd
    rgba = np.random.default_rng(0).integers(0, 255, (h, w, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    return bytes(hdr) + dtx._encode_pixels(rgba, bpp), rgba


def test_parse_and_decode_dxt1():
    raw, _ = _fake()
    d = dtx.parse(raw)
    assert (d.width, d.height, d.mips, d.format) == (64, 64, 1, "DXT1")
    assert dtx.decode(d).shape == (64, 64, 4)


def test_write_keeps_header_fields_and_changes_size():
    raw, rgba = _fake()
    d = dtx.parse(raw)
    big = np.kron(dtx.decode(d), np.ones((2, 2, 1), np.uint8))
    out = dtx.write(d, big)
    p = dtx.parse(out)
    assert (p.width, p.height) == (128, 128)
    assert p.command == "alpharef 128" and p.iflags == d.iflags
    assert out[16:36] == raw[16:36]   # flags, user flags, group/priority extras
    assert out[36:164] == raw[36:164]  # command string
    assert len(out) == 164 + dtx.mip_size(128, 128, dtx.BPP_DXT1)


def test_extra_mip_count_mirrors_header():
    raw, _ = _fake(bpp=dtx.BPP_DXT5)
    d = dtx.parse(raw)
    out = dtx.write(d, dtx.decode(d), mips=3)
    p = dtx.parse(out)
    assert p.mips == 3 and out[25] == 3
