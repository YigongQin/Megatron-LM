# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

_HAVE_FLASHINFER_MOE_EP = False
_FLASHINFER_MOE_EP_IMPORT_ERROR: ImportError | None = None

try:
    import flashinfer.moe_ep.backends  # noqa: F401
    from flashinfer.moe_ep import BootstrapConfig, FleetParams, MegaConfig, MoEEpMegaLayer
    from flashinfer.moe_ep.tensors import MoEEpTensors

    _HAVE_FLASHINFER_MOE_EP = True
except ImportError as exc:
    BootstrapConfig = None  # type: ignore[misc, assignment]
    FleetParams = None  # type: ignore[misc, assignment]
    MegaConfig = None  # type: ignore[misc, assignment]
    MoEEpMegaLayer = None  # type: ignore[misc, assignment]
    MoEEpTensors = None  # type: ignore[misc, assignment]
    _FLASHINFER_MOE_EP_IMPORT_ERROR = exc


def require_flashinfer_moe_ep() -> None:
    if not _HAVE_FLASHINFER_MOE_EP:
        raise RuntimeError(
            "inference_grouped_gemm_backend='flashinfer_mega' requires flashinfer-python "
            "with moe_ep mega kernels (flashinfer >= 0.6.x, NVSHMEM build for CuTeDSL mega). "
            "Install or upgrade flashinfer-python."
        ) from _FLASHINFER_MOE_EP_IMPORT_ERROR
