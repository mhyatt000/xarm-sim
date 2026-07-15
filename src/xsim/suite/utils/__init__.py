import genesis as gs

from xsim.suite.utils.placement_samplers import UniformRandomSampler

__all__ = ["UniformRandomSampler", "ensure_genesis_init"]


def ensure_genesis_init(**kwargs) -> None:
    """Initialize Genesis unless the caller already did (gs.init raises on re-init)."""
    if not gs._initialized:
        gs.init(**kwargs)
