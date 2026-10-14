"""Replace files inside a LithTech `.rez` archive (Combat Arms BR client) without touching the other files.

Format (from ENGINE/libs/rezmgr/rezmgr.cpp, and checked against every shipped archive by tools/rez.py):
    0x00..0xC2  header text and the encode/detect block (a self-consistent integrity pair; kept verbatim)
    0xC3        10 x u32: version, root dir pos, root dir size, root dir time, next write pos (= end of data /
                start of the directories), file time, largest key array, largest dir name, largest rez name,
                largest comment;  then u8 is-sorted
    data        item payloads, written in place
    dirs        each directory block is a list of entries:
                  u32 0 (item): pos,size,time,id,type,nKeys (u32 each), name\\0, comment\\0, keys[nKeys] u32
                  u32 1 (dir) : pos,size,time (u32 each), name\\0   (pos/size point at that directory's block)

``replace`` keeps the original file up to ``next write pos``, appends the new payloads there, then writes fresh
directory blocks after them and updates the header numbers. Unchanged items keep their exact bytes and positions.
The old payloads of replaced items stay in the file as unreferenced bytes (like an editor's append-only save).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

HDR = 0xC3


@dataclass
class Item:
    name: str
    pos: int
    size: int
    time: int
    id: int
    type: int
    comment: str
    keys: list[int]


@dataclass
class Dir:
    name: str
    time: int
    entries: list = field(default_factory=list)  # Item or Dir, original order


def _cstr(b, p):
    e = b.index(b"\0", p)
    return b[p:e].decode("latin1"), e + 1


def _read_dir(b: bytes, pos: int, size: int, name: str, time: int) -> Dir:
    d = Dir(name, time)
    blk = b[pos:pos + size]
    p = 0
    while p < len(blk):
        t, = struct.unpack_from("<I", blk, p); p += 4
        if t == 1:
            dpos, dsize, dtime = struct.unpack_from("<3I", blk, p); p += 12
            n, p = _cstr(blk, p)
            d.entries.append(_read_dir(b, dpos, dsize, n, dtime))
        else:
            ipos, isize, itime, iid, ityp, nk = struct.unpack_from("<6I", blk, p); p += 24
            n, p = _cstr(blk, p)
            c, p = _cstr(blk, p)
            keys = list(struct.unpack_from(f"<{nk}I", blk, p)); p += 4 * nk
            d.entries.append(Item(n, ipos, isize, itime, iid, ityp, c, keys))
    return d


def parse(b: bytes):
    ver, rpos, rsize, rtime, nextw, tm, lka, lds, lrs, lcs = struct.unpack_from("<10I", b, HDR)
    return _read_dir(b, rpos, rsize, "", rtime), nextw


def walk(d: Dir, prefix=""):
    for e in d.entries:
        if isinstance(e, Dir):
            yield from walk(e, prefix + e.name + "/")
        else:
            yield prefix + e.name, e


def _write_dir(d: Dir, out: bytearray, base: int) -> tuple[int, int]:
    """Serialize children first, then this block. Returns (pos, size)."""
    child = {}
    for e in d.entries:
        if isinstance(e, Dir):
            child[id(e)] = _write_dir(e, out, base)
    blk = bytearray()
    for e in d.entries:
        if isinstance(e, Dir):
            pos, size = child[id(e)]
            blk += struct.pack("<4I", 1, pos, size, e.time) + e.name.encode("latin1") + b"\0"
        else:
            blk += struct.pack("<7I", 0, e.pos, e.size, e.time, e.id, e.type, len(e.keys))
            blk += e.name.encode("latin1") + b"\0" + e.comment.encode("latin1") + b"\0"
            blk += struct.pack(f"<{len(e.keys)}I", *e.keys)
    pos = base + len(out)
    out += blk
    return pos, len(blk)


def replace(b: bytes, new: dict[str, bytes]) -> bytes:
    """Return a new archive in which each path in ``new`` (as listed by ``walk``) has the given payload."""
    root, nextw = parse(b)
    found = {}
    for path, it in walk(root):
        if path in new:
            found[path] = it
    missing = set(new) - set(found)
    if missing:
        raise KeyError(f"not in archive: {sorted(missing)}")
    out = bytearray()
    for path, it in found.items():           # append payloads where the directories used to start
        it.pos, it.size = nextw + len(out), len(new[path])
        out += new[path]
    dirs_at = nextw + len(out)
    dirs = bytearray()
    rpos, rsize = _write_dir(root, dirs, dirs_at)
    head = bytearray(b[:nextw])
    struct.pack_into("<I", head, HDR + 4, rpos)        # root dir pos
    struct.pack_into("<I", head, HDR + 8, rsize)       # root dir size
    struct.pack_into("<I", head, HDR + 16, dirs_at)    # next write pos
    return bytes(head) + bytes(out) + bytes(dirs)
