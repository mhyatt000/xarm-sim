"""Training-side code: networks, collection, and distributed mechanics.

The suite stays env-only; anything that learns lives here (or, for the
vendored GC/tdmpc agents, in xsim.suite.algo pending migration).
"""

from __future__ import annotations

from xsim.algo.dagger import (
    Collector,
    MLPTeacher,
    aux_targets,
    flat,
    image_proprio_keys,
)
from xsim.algo.distributed import Distributed
from xsim.algo.fk import FKChain
from xsim.algo.nets import (
    FlowImageStudent,
    FlowStateStudent,
    ImageStudent,
    Student,
    rand_shift,
    time_features,
)

__all__ = [
    "Collector",
    "Distributed",
    "FKChain",
    "FlowImageStudent",
    "FlowStateStudent",
    "ImageStudent",
    "MLPTeacher",
    "Student",
    "aux_targets",
    "flat",
    "image_proprio_keys",
    "rand_shift",
    "time_features",
]
