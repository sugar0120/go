"""LithTech DTX (version -5) reader and writer for the Combat Arms BR client.

Layout, measured on all 17,631 shipped textures (164-byte header):

    0   u32  resource type (0)
    4   u16  width        6  u16 height
    8   i32  version (-5)
    12  u16  mip count    14 u16 section count
    16  u32  IFlags       20 u32 UserFlags
    24  u8[12] extra: [0] texture group, [1] mip count, [2] BPP id,
                      [3] non-S3TC mip offset, [4] UI mip offset, [5] priority
    36  char[128] command string ("alpharef 128", "envmapalpha <path>", ...)
    164 pixel data, largest mip first, then (if sections) section data

BPP id: 2 = 16-bit 565, 3 = BGRA32, 4 = DXT1, 5 = DXT3, 6 = DXT5.

``write`` copies the original header byte for byte and only changes the size
and mip fields, so flags, group, priority and the command string survive.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

HEADER_SIZE = 164
BPP_16, BPP_BGRA32, BPP_DXT1, BPP_DXT3, BPP_DXT5 = 2, 3, 4, 5, 6
FORMAT_NAMES = {BPP_16: "RGB565", BPP_BGRA32: "BGRA32", BPP_DXT1: "DXT1", BPP_DXT3: "DXT3", BPP_DXT5: "DXT5"}


@dataclass
class Dtx:
    header: bytes          # original 164 header bytes (kept verbatim)
    width: int
    height: int
    mips: int
    bpp: int
    command: str
    iflags: int
    trailer: bytes         # anything after the pixel data (sections), kept verbatim
    mip_data: list[bytes]  # raw bytes per mip, largest first

    @property
    def format(self) -> str:
        return FORMAT_NAMES.get(self.bpp, str(self.bpp))


def mip_size(w: int, h: int, bpp: int) -> int:
    if bpp == BPP_DXT1:
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 8
    if bpp in (BPP_DXT3, BPP_DXT5):
        return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * 16
    if bpp == BPP_BGRA32:
        return w * h * 4
    if bpp == BPP_16:
        return w * h * 2
    raise ValueError(f"unsupported BPP id {bpp}")


def parse(data: bytes) -> Dtx:
    if len(data) < HEADER_SIZE:
        raise ValueError("too short for a DTX header")
    rtype, = struct.unpack_from("<I", data, 0)
    w, h = struct.unpack_from("<HH", data, 4)
    ver, = struct.unpack_from("<i", data, 8)
    mips, nsec = struct.unpack_from("<HH", data, 12)
    iflags, = struct.unpack_from("<I", data, 16)
    bpp = data[0x1A] or BPP_BGRA32
    if rtype != 0 or ver != -5 or not (0 < w <= 8192 and 0 < h <= 8192) or not (1 <= mips <= 15):
        raise ValueError(f"not a version -5 DTX (type={rtype} ver={ver} {w}x{h} mips={mips})")
    cmd = data[36:164].split(b"\0")[0].decode("latin1")
    pos, mw, mh, mip_data = HEADER_SIZE, w, h, []
    for _ in range(mips):
        n = mip_size(mw, mh, bpp)
        mip_data.append(data[pos:pos + n])
        pos += n
        mw, mh = max(1, mw // 2), max(1, mh // 2)
    if len(mip_data[-1]) != mip_size(max(1, w >> (mips - 1)), max(1, h >> (mips - 1)), bpp):
        raise ValueError("truncated pixel data")
    return Dtx(bytes(data[:HEADER_SIZE]), w, h, mips, bpp, cmd, iflags, bytes(data[pos:]), mip_data)


def decode(dtx: Dtx, mip: int = 0) -> np.ndarray:
    """Decode one mip to an HxWx4 uint8 RGBA array."""
    w, h = max(1, dtx.width >> mip), max(1, dtx.height >> mip)
    raw = dtx.mip_data[mip]
    if dtx.bpp == BPP_BGRA32:
        a = np.frombuffer(raw, np.uint8).reshape(h, w, 4)
        return a[..., [2, 1, 0, 3]].copy()
    if dtx.bpp == BPP_16:
        v = np.frombuffer(raw, "<u2").reshape(h, w).astype(np.uint32)
        r, g, b = (v >> 11 & 31) * 255 // 31, (v >> 5 & 63) * 255 // 63, (v & 31) * 255 // 31
        return np.stack([r, g, b, np.full_like(r, 255)], -1).astype(np.uint8)
    if dtx.bpp in (BPP_DXT1, BPP_DXT3, BPP_DXT5):
        return _decode_s3tc(raw, w, h, dtx.bpp)
    raise NotImplementedError(f"decoding {dtx.format} is not supported")


def _decode_s3tc(raw: bytes, w: int, h: int, bpp: int) -> np.ndarray:
    """Decode DXT1/3/5 with Pillow's DDS reader (etcpak's decoder is unreliable)."""
    import io
    from PIL import Image

    fourcc = {BPP_DXT1: b"DXT1", BPP_DXT3: b"DXT3", BPP_DXT5: b"DXT5"}[bpp]
    hdr = (b"DDS " + struct.pack("<7I", 124, 0x1007, h, w, 0, 0, 1) + b"\0" * 44
           + struct.pack("<2I4s5I", 32, 4, fourcc, 0, 0, 0, 0, 0) + struct.pack("<5I", 0x1000, 0, 0, 0, 0))
    return np.asarray(Image.open(io.BytesIO(hdr + raw)).convert("RGBA")).copy()


def _encode_pixels(rgba: np.ndarray, bpp: int) -> bytes:
    h, w = rgba.shape[:2]
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    if bpp == BPP_BGRA32:
        return rgba[..., [2, 1, 0, 3]].tobytes()
    import etcpak
    if bpp == BPP_DXT1:
        return bytes(etcpak.compress_bc1(rgba.tobytes(), w, h))
    if bpp == BPP_DXT5:
        return bytes(etcpak.compress_bc3(rgba.tobytes(), w, h))
    raise NotImplementedError(f"encoding {FORMAT_NAMES.get(bpp, bpp)} is not supported")


def write(original: Dtx, rgba: np.ndarray, mips: int | None = None) -> bytes:
    """Encode ``rgba`` (any power-of-two size) using the original's format and header.

    ``mips`` defaults to the original mip count (weapon textures have 1).
    """
    from PIL import Image

    h, w = rgba.shape[:2]
    n = mips or original.mips
    hdr = bytearray(original.header)
    struct.pack_into("<HH", hdr, 4, w, h)
    struct.pack_into("<H", hdr, 12, n)
    hdr[25] = n  # Extra[1] mirrors the mip count
    body, cur = [], rgba
    for i in range(n):
        body.append(_encode_pixels(cur, original.bpp))
        if i + 1 < n:
            im = Image.fromarray(cur, "RGBA").resize((max(1, cur.shape[1] // 2), max(1, cur.shape[0] // 2)), Image.BOX)
            cur = np.asarray(im)
    return bytes(hdr) + b"".join(body) + original.trailer
