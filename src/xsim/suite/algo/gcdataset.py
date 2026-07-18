"""Numpy port of ogbench's GCDataset goal relabeling (ogbench/utils/datasets.py).

Stores flat (transition-major) trajectories and samples batches with relabeled
``value_goals`` / ``actor_goals`` plus the ``actor_offsets`` used by CFGRL's
step conditioning. Frame stacking, image augmentation, oracle reps, and the
horizon>1 / hierarchical variants are not ported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GCDatasetConfig:
    """Goal-sampling hyperparameters shared by the goal-conditioned agents."""

    discount: float = 0.99  # geometric goal-sampling decay
    value_p_curgoal: float = 0.0  # P(value goal = current state)
    value_p_trajgoal: float = 1.0  # P(value goal = future state, same traj)
    value_p_randomgoal: float = 0.0  # P(value goal = random state)
    value_geom_sample: bool = False  # geometric (vs uniform) future value goals
    actor_p_curgoal: float = 0.0  # P(actor goal = current state)
    actor_p_trajgoal: float = 1.0  # P(actor goal = future state, same traj)
    actor_p_randomgoal: float = 0.0  # P(actor goal = random state)
    actor_geom_sample: bool = False  # geometric (vs uniform) future actor goals
    gc_negative: bool = True  # reward = -1/0 (True) or 0/1 (False)


class GCDataset:
    """Goal-relabeling sampler over a flat transition dataset.

    ``dataset`` maps keys to (N, ...) arrays and must contain ``observations``,
    ``actions``, and ``terminals`` (1 at the last step of each trajectory; the
    final transition of the dataset must be terminal).
    """

    def __init__(self, dataset: dict[str, np.ndarray], config: GCDatasetConfig):
        self.dataset = dataset
        self.config = config
        self.size = len(dataset["observations"])

        (self.terminal_locs,) = np.nonzero(dataset["terminals"] > 0)
        assert self.terminal_locs[-1] == self.size - 1

        assert np.isclose(config.value_p_curgoal + config.value_p_trajgoal + config.value_p_randomgoal, 1.0)
        assert np.isclose(config.actor_p_curgoal + config.actor_p_trajgoal + config.actor_p_randomgoal, 1.0)

    def sample(self, batch_size: int, idxs: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Sample a batch of transitions with relabeled goals."""
        if idxs is None:
            idxs = np.random.randint(self.size, size=batch_size)

        batch = {k: v[idxs] for k, v in self.dataset.items()}

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config.value_p_curgoal,
            self.config.value_p_trajgoal,
            self.config.value_p_randomgoal,
            self.config.value_geom_sample,
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config.actor_p_curgoal,
            self.config.actor_p_trajgoal,
            self.config.actor_p_randomgoal,
            self.config.actor_geom_sample,
        )

        batch["value_goals"] = self.dataset["observations"][value_goal_idxs]
        batch["actor_goals"] = self.dataset["observations"][actor_goal_idxs]
        batch["actor_offsets"] = (actor_goal_idxs - idxs).astype(np.int64)[:, None]

        successes = (idxs == value_goal_idxs).astype(np.float32)
        batch["masks"] = 1.0 - successes
        batch["rewards"] = successes - (1.0 if self.config.gc_negative else 0.0)

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample, discount=None):
        """Sample goal indices for the given transition indices."""
        batch_size = len(idxs)
        if discount is None:
            discount = self.config.discount

        random_goal_idxs = np.random.randint(self.size, size=batch_size)

        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            offsets = np.random.geometric(p=1 - discount, size=batch_size)  # in [1, inf)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)  # in [0, 1)
            traj_goal_idxs = np.round(
                np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances)
            ).astype(int)

        if p_curgoal == 1.0:
            goal_idxs = idxs
        else:
            goal_idxs = np.where(
                np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal), traj_goal_idxs, random_goal_idxs
            )
            goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs


def to_torch(batch: dict[str, np.ndarray], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Convert a sampled batch to torch tensors on ``device``."""
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
