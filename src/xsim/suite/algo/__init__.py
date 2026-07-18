"""Vendored learning algorithms.

The tdmpc2 checkout uses top-level absolute imports internally (``from common
import math``), so it cannot be imported as a subpackage; importing
``xsim.suite.algo`` puts its package root on ``sys.path`` instead:

    import xsim.suite.algo  # noqa: F401
    from tdmpc2 import TDMPC2
    from common.buffer import Buffer
"""

from __future__ import annotations

import sys
from pathlib import Path

from xsim.suite.algo.cfgrl import CFGRLAgent, CFGRLConfig
from xsim.suite.algo.gcbc import GCBCAgent, GCBCConfig
from xsim.suite.algo.gcdataset import GCDataset, GCDatasetConfig, to_torch
from xsim.suite.algo.gcfbc import GCFBCAgent, GCFBCConfig

_TDMPC2_ROOT = str(Path(__file__).resolve().parent / "tdmpc2" / "tdmpc2")
if _TDMPC2_ROOT not in sys.path:
    sys.path.insert(0, _TDMPC2_ROOT)

__all__ = [
    "CFGRLAgent",
    "CFGRLConfig",
    "GCBCAgent",
    "GCBCConfig",
    "GCDataset",
    "GCDatasetConfig",
    "GCFBCAgent",
    "GCFBCConfig",
    "to_torch",
]
