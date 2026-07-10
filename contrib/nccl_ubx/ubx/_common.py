"""Hardware detection helpers."""

import torch


def _get_cuda_sm_version() -> tuple[int, int]:
    """Get the SM version (major, minor) of the current CUDA device."""
    if not torch.cuda.is_available():
        return (0, 0)
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor)
