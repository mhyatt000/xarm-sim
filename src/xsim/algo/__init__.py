"""Training-side code: networks, collection, and distributed mechanics.

The suite stays env-only; anything that learns lives here (or, for the
vendored GC/tdmpc agents, in xsim.suite.algo pending migration).
"""

from __future__ import annotations

from xsim.algo.augs import AugmentedDataset, sim2real_transform
from xsim.algo.dagger import (
    Collector,
    MLPTeacher,
    aux_targets,
    flat,
    image_proprio_keys,
)
from xsim.algo.distributed import Distributed
from xsim.algo.nets import (
    FlowImageStudent,
    ImageStudent,
    Student,
    rand_shift,
    time_features,
)

__all__ = [
    "AugmentedDataset",
    "Collector",
    "Distributed",
    "FlowImageStudent",
    "ImageStudent",
    "MLPTeacher",
    "Student",
    "aux_targets",
    "flat",
    "image_proprio_keys",
    "rand_shift",
    "sim2real_transform",
    "time_features",
]
