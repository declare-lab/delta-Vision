"""delta-vision package.

Top-level exports are lazy so lightweight CLI modules do not import the model
stack unless they actually need it.
"""

__all__ = [
    "BasisCoefficientSidecar",
    "DeltaVisionModule",
    "DeltaVisionModel",
    "SidecarRolloutModel",
    "VisualKVCache",
]


def __getattr__(name: str):
    if name in {"BasisCoefficientSidecar", "DeltaVisionModule"}:
        from delta_vision.models.sidecar import BasisCoefficientSidecar, DeltaVisionModule

        return {"BasisCoefficientSidecar": BasisCoefficientSidecar, "DeltaVisionModule": DeltaVisionModule}[name]
    if name in {"DeltaVisionModel", "SidecarRolloutModel"}:
        from delta_vision.models.modeling import DeltaVisionModel, SidecarRolloutModel

        return {"DeltaVisionModel": DeltaVisionModel, "SidecarRolloutModel": SidecarRolloutModel}[name]
    if name == "VisualKVCache":
        from delta_vision.runtime.ops import VisualKVCache

        return VisualKVCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
