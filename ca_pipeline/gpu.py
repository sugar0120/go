"""GPU sanity checks. molab's GPU is an RTX Pro 6000 Blackwell (96 GB)."""

from __future__ import annotations


def blackwell_issues(capability: tuple[int, int], torch_cuda: str | None,
                     arch_list: list[str]) -> list[str]:
    """Problems that stop PyTorch from using a Blackwell card."""
    issues = []
    if capability[0] < 10:
        return issues  # not Blackwell, nothing special to check
    if torch_cuda is None:
        issues.append("PyTorch is a CPU-only build; install a CUDA 12.8+ build.")
        return issues
    major, minor = (int(x) for x in torch_cuda.split(".")[:2])
    if (major, minor) < (12, 8):
        issues.append(f"PyTorch built for CUDA {torch_cuda}; Blackwell needs 12.8+ (torch 2.7+, cu128 wheels).")
    sm = f"sm_{capability[0]}{capability[1]}"
    if arch_list and sm not in arch_list:
        issues.append(f"This PyTorch build has no kernels for {sm} (has {', '.join(arch_list)}).")
    return issues


def describe() -> dict:
    """Name, memory and readiness of the current GPU (empty name if none)."""
    try:
        import torch
    except ImportError:
        return {"name": "", "vram_gib": 0.0, "issues": ["torch not installed"]}
    if not torch.cuda.is_available():
        return {"name": "", "vram_gib": 0.0, "issues": []}
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    return {
        "name": props.name,
        "vram_gib": props.total_memory / 2**30,
        "capability": f"{cap[0]}.{cap[1]}",
        "bf16": torch.cuda.is_bf16_supported(),
        "issues": blackwell_issues(cap, torch.version.cuda, torch.cuda.get_arch_list()),
    }
