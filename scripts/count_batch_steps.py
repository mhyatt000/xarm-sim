"""Count recorded timesteps in a batch of MCAP episodes.

By default, one "step" means one message on /xarm/joint_states. For the sim
lift batches, core topics are logged in lockstep, so this is the recorded
training timestep count. Use --topic to count a different stream.

    uv run python scripts/count_batch_steps.py /data/store/griffen_sim_mcaps/<batch>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_mcap import counts_by_topic, scan

DEFAULT_TOPIC = "/xarm/joint_states"


def count_steps(path: Path, topic: str) -> int:
    chans, _, counts, _ = scan(str(path))
    return counts_by_topic(chans, counts).get(topic, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch_dir", type=Path, help="Directory containing episode_*.mcap files")
    ap.add_argument("--topic", default=DEFAULT_TOPIC, help=f"Topic to count; default: {DEFAULT_TOPIC}")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = ap.parse_args()

    files = sorted(args.batch_dir.glob("episode_*.mcap"))
    if not files:
        raise SystemExit(f"no episode_*.mcap files found in {args.batch_dir}")

    per_episode = [{"file": f.name, "steps": count_steps(f, args.topic)} for f in files]
    steps = [e["steps"] for e in per_episode]
    summary = {
        "batch_dir": str(args.batch_dir),
        "topic": args.topic,
        "episodes": len(per_episode),
        "total_steps": sum(steps),
        "min_steps": min(steps),
        "max_steps": max(steps),
        "mean_steps": sum(steps) / len(steps),
        "per_episode": per_episode,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"batch: {summary['batch_dir']}")
    print(f"topic: {summary['topic']}")
    print(f"episodes: {summary['episodes']}")
    print(f"total steps: {summary['total_steps']}")
    print(
        f"steps/episode: min={summary['min_steps']} "
        f"mean={summary['mean_steps']:.2f} max={summary['max_steps']}"
    )


if __name__ == "__main__":
    main()
