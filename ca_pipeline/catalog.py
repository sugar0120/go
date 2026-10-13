"""Candidate upscaling models, smallest to largest.

"Biggest" is not automatically "best" for game textures: large transformer
models trained on clean photo downscales can over-smooth DXT-compressed
textures, while smaller GAN models trained on degraded inputs often look
sharper. Use notebooks/02_benchmark_upscalers.py on the pilot set to decide.

Download weights from OpenModelDB (https://openmodeldb.info) or the project
repos named below into ./models/ (git-ignored). Parameter counts are
approximate and only meant to rank VRAM/speed cost.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    key: str
    arch: str
    params_millions: float
    source: str
    notes: str
    via_spandrel: bool = True


CANDIDATES: list[Candidate] = [
    Candidate("lanczos", "resample", 0, "built in",
              "Baseline. Everything must beat this."),
    Candidate("realesrgan-x4plus", "ESRGAN (RRDB)", 16.7, "github.com/xinntao/Real-ESRGAN",
              "General real-world degradations. Good first AI baseline."),
    Candidate("4x-ultrasharp", "ESRGAN (RRDB)", 16.7, "OpenModelDB",
              "Popular for game textures; sharp, handles compression artifacts."),
    Candidate("4x-remacri", "ESRGAN (RRDB)", 16.7, "OpenModelDB",
              "Softer, more natural detail; good for skin/fabric."),
    Candidate("swinir-l-realworld", "SwinIR-L", 11.9, "github.com/JingyunLiang/SwinIR",
              "Transformer, real-world GAN variant. Slower than ESRGAN."),
    Candidate("dat-2", "DAT", 11.2, "github.com/zhengchen1999/DAT",
              "Transformer; very clean output, can look over-smoothed."),
    Candidate("drct-l", "DRCT-L", 27.6, "github.com/ming053l/DRCT",
              "Large transformer; strong fine detail, heavy VRAM."),
    Candidate("hat-l", "HAT-L", 40.8, "github.com/XPixelGroup/HAT",
              "Largest spandrel-supported option here. Use the Real_HAT_GAN "
              "weights for compressed textures; the classical SRx4 weights "
              "expect clean bicubic input."),
    Candidate("supir", "SUPIR (SDXL diffusion)", 2600, "github.com/Fanghua-Yu/SUPIR",
              "Diffusion upscaler: invents new detail, may change the art. "
              "Needs ~24 GB+ VRAM and its own install; hero assets only, "
              "always review by hand.", via_spandrel=False),
]


def recommend_for_vram(vram_gib: float) -> list[str]:
    """Which candidates are practical on a GPU with ``vram_gib`` of memory."""
    if vram_gib <= 0:
        return ["lanczos", "realesrgan-x4plus"]  # CPU: only small models are tolerable
    keys = [c.key for c in CANDIDATES if c.via_spandrel]
    if vram_gib >= 24:
        keys.append("supir")
    return keys
