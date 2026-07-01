"""Validate a synthetic episode MCAP against the reference `_base.mcap` schema.

Reads an MCAP with only numpy/opencv/numcodecs (no mcap/foxglove reader needed):
lists channels + schemas, counts messages per channel, decodes the first RawImage of each
camera (dims/encoding/frame_id + YUYV sanity), and checks the channel set is a superset of
a reference file's image channels.

Usage:
    python scripts/validate_mcap.py outputs/sim_mcap/fake.mcap \
        --reference /data/fast/episodes/260618-122602_000007_base.mcap
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct

import cv2
import numpy as np
from numcodecs import Zstd

REFERENCE_IMAGE_TOPICS = {"/cam/side/image_raw", "/cam/wrist/image_raw", "/cam/over/image_raw"}


def _str(b: bytes, o: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<I", b, o)
    o += 4
    return b[o : o + n].decode("utf-8", "replace"), o + n


def read_records(path: str):
    """Yield (op, body) for every top-level record, transparently unchunking (zstd/none)."""
    with open(path, "rb") as f:
        f.seek(8)  # skip magic
        data = f.read()
    o = 0
    while o < len(data) - 8:
        op = data[o]
        (ln,) = struct.unpack_from("<Q", data, o + 1)
        body = data[o + 9 : o + 9 + ln]
        o += 9 + ln
        if op == 0x02:  # footer
            break
        if op == 0x06:  # chunk
            p = 28
            comp, p = _str(body, p)
            (cs,) = struct.unpack_from("<Q", body, p)
            p += 8
            rec = Zstd().decode(body[p : p + cs]) if comp == "zstd" else body[p : p + cs]
            q = 0
            while q < len(rec):
                rop = rec[q]
                (rln,) = struct.unpack_from("<Q", rec, q + 1)
                yield rop, rec[q + 9 : q + 9 + rln]
                q += 9 + rln
        else:
            yield op, body


def scan(path: str):
    chans: dict[int, tuple[str, int]] = {}
    schemas: dict[int, str] = {}
    counts: dict[int, int] = {}
    first_msg: dict[int, bytes] = {}
    for op, body in read_records(path):
        if op == 0x03:  # schema
            (sid,) = struct.unpack_from("<H", body, 0)
            name, _ = _str(body, 2)
            schemas[sid] = name
        elif op == 0x04:  # channel
            (cid, sid) = struct.unpack_from("<HH", body, 0)
            topic, _ = _str(body, 4)
            chans[cid] = (topic, sid)
        elif op == 0x05:  # message
            (cid,) = struct.unpack_from("<H", body, 0)
            counts[cid] = counts.get(cid, 0) + 1
            first_msg.setdefault(cid, body[22:])
    return chans, schemas, counts, first_msg


def _varint(b: bytes, o: int) -> tuple[int, int]:
    v = s = 0
    while True:
        x = b[o]
        o += 1
        v |= (x & 0x7F) << s
        s += 7
        if not x & 0x80:
            return v, o


def parse_rawimage(buf: bytes) -> dict:
    """Foxglove RawImage: f2=width(fixed32), f3=height, f4=encoding, f5=step, f6=data, f7=frame_id."""
    o = 0
    out: dict = {}
    while o < len(buf):
        tag, o = _varint(buf, o)
        fn, wt = tag >> 3, tag & 7
        if wt == 0:
            out[fn], o = _varint(buf, o)
        elif wt == 2:
            n, o = _varint(buf, o)
            out[fn] = buf[o : o + n]
            o += n
        elif wt == 1:
            o += 8
        elif wt == 5:
            (out[fn],) = struct.unpack_from("<I", buf, o)
            o += 4
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mcap", type=Path)
    ap.add_argument("--reference", type=Path, default=None)
    args = ap.parse_args()

    chans, schemas, counts, first = scan(str(args.mcap))
    by_topic = {t: (cid, sid) for cid, (t, sid) in chans.items()}

    print(f"== {args.mcap} ==")
    print(f"{'topic':30s} {'schema':28s} count")
    for cid, (topic, sid) in sorted(chans.items(), key=lambda kv: kv[1][0]):
        print(f"{topic:30s} {schemas.get(sid, '?'):28s} {counts.get(cid, 0)}")

    print("\n-- RawImage decode --")
    for topic in sorted(t for t in by_topic if t.endswith("/image_raw")):
        cid, _ = by_topic[topic]
        ri = parse_rawimage(first[cid])
        w, h, step = ri.get(2), ri.get(3), ri.get(5)
        enc = ri[4].decode() if 4 in ri else "?"
        fid = ri[7].decode() if 7 in ri else "?"
        ok = "OK" if (enc == "yuv422_yuy2" and step == (w or 0) * 2 and len(ri.get(6, b"")) == (w or 0) * (h or 0) * 2) else "??"
        yuyv = np.frombuffer(ri[6], np.uint8).reshape(h, w, 2)
        rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUY2)
        print(f"  [{ok}] {topic:24s} {w}x{h} step={step} enc={enc} frame_id={fid!r} rgb_mean={rgb.mean():.0f}")

    if args.reference and args.reference.exists():
        ref_chans, *_ = scan(str(args.reference))
        ref_topics = {t for _, (t, _) in ref_chans.items()}
        img_ref = ref_topics & REFERENCE_IMAGE_TOPICS
        have = set(by_topic)
        missing = img_ref - have
        print(f"\n-- vs reference {args.reference.name} --")
        print(f"  reference image topics: {sorted(ref_topics)}")
        print(f"  superset of reference image channels: {'YES' if not missing else f'MISSING {missing}'}")


if __name__ == "__main__":
    main()
