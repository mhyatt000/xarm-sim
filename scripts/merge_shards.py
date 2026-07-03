"""Merge sharded batch outputs into one flat batch dir (batch_v2 manifest schema).

Each shard dir holds episode_%06d.mcap + manifest.json from one generator run.
Episodes are renumbered sequentially in shard order, hard-linked into the batch
root, and the manifests are merged: config comes from shard 0 (n_episodes and
seed updated), every episode entry keeps its own seed and gains a "shard" field,
and the splat identity must agree across shards.

    uv run python scripts/merge_shards.py --batch-dir outputs/sim_mcap/batch_v3
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Cfg:
    batch_dir: Path
    shard_glob: str = "shard_*"
    note: str = ""


def main(c: Cfg) -> None:
    shards = sorted(c.batch_dir.glob(c.shard_glob))
    if not shards:
        raise SystemExit(f"no shards matching {c.shard_glob} in {c.batch_dir}")
    manifests = [json.loads((s / "manifest.json").read_text()) for s in shards]

    splats = {(m["splat_file"], m["splat_md5"]) for m in manifests}
    if len(splats) != 1:
        raise SystemExit(f"shards disagree on splat identity: {splats}")
    shas = {m["git_sha"] for m in manifests}
    if len(shas) != 1:
        print(f"WARNING: shards built from different commits: {shas}")

    merged = dict(manifests[0])
    merged["episodes"] = []
    out_idx = 0
    for k, (shard, m) in enumerate(zip(shards, manifests)):
        for ep in m["episodes"]:
            src = shard / f"episode_{ep['episode']:06d}.mcap"
            dst = c.batch_dir / f"episode_{out_idx:06d}.mcap"
            if dst.exists():
                dst.unlink()
            os.link(src, dst)  # hard link: no copy, shard dirs stay intact
            entry = dict(ep)
            entry["episode"] = out_idx
            entry["shard"] = k
            merged["episodes"].append(entry)
            out_idx += 1

    merged["config"]["n_episodes"] = out_idx
    merged["config"]["out_dir"] = str(c.batch_dir)
    n_ok = sum(e["success"] for e in merged["episodes"])
    merged["success_rate"] = n_ok / out_idx
    if c.note:
        merged["note"] = c.note
    (c.batch_dir / "manifest.json").write_text(json.dumps(merged, indent=1))
    print(f"merged {out_idx} episodes from {len(shards)} shards "
          f"({n_ok}/{out_idx} success) -> {c.batch_dir}/manifest.json")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
