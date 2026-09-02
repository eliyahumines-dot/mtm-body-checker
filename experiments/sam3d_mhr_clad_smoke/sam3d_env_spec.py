"""Environment A (SAM 3D Body core inference) dependency specification --
Task 03C.

Single source of truth for what Phase A installs, imported by both the
Colab notebook (so the pip command it actually runs is built from this,
not a separately hand-typed string that can drift) and this module's own
test suite (so a future edit that reintroduces an excluded package fails a
fast, GPU-free unit test instead of surfacing as a wheel-build error deep
into a real Colab run, the way it did in Task 03B).

## chump vs. chumpy

Task 03B's notebook installed `chumpy` (0.70, PyPI) -- the well-known
SMPL-adjacent autodiff library many body-mesh projects use, which is a
reasonable guess but was never actually verified against the primary
source. It failed to build a wheel in the real Colab run.

Verified directly against a fresh clone of `facebookresearch/sam-3d-body`
(commit `b5c765a`, INSTALL.md) for this task: the official dependency list
reads `... pyrootutils webdataset chump networkx==3.2.1 roma ...` --
**`chump`** (a small, unrelated PyPI package, `pip index versions chump`
-> 1.6.0), not `chumpy`. This was a transcription mistake introduced when
Task 03/03B's notebook was authored, not an upstream naming change. A
full-repo grep of `sam-3d-body`'s own Python source for `chump`/`chumpy`
found zero matches either way -- neither name is `import`-ed directly
anywhere in the repo, so whichever package genuinely needs it is pulled in
transitively by one of the *other* listed dependencies, not by
`sam_3d_body` itself; this module does not attempt to identify which one,
since Task 03C's minimal list already matches upstream's own literal text
exactly and introduces no new uncertainty there.
"""

from __future__ import annotations

TORCH_PIN = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "index_url": "https://download.pytorch.org/whl/cu129",
}

# Exactly SAM 3D Body's own INSTALL.md pip list, MINUS:
#   - platform-irrelevant/demo-only extras this project doesn't use here
#     (appnope is macOS-only, ffmpeg/cython/pytest/black/tensorboard are not
#     needed for a single inference call), and
#   - everything Task 03C explicitly excludes from the minimal core-inference
#     smoke test: no Detectron2 (no learned detector -- explicit full-image
#     bbox instead), no MoGe (no FOV estimator -- upstream's own documented
#     default-FOV fallback), no SAM2/SAM3 (no segmentor).
# "chump" (not "chumpy" -- see module docstring) is the one entry most worth
# double-checking on sight, which is exactly why it's data here, not a
# string embedded directly in the notebook.
SAM3D_CORE_PIP_DEPENDENCIES: tuple[str, ...] = (
    "pytorch-lightning",
    "pyrender",
    "opencv-python",
    "yacs",
    "scikit-image",
    "einops",
    "timm",
    "dill",
    "pandas",
    "rich",
    "hydra-core",
    "hydra-submitit-launcher",
    "hydra-colorlog",
    "pyrootutils",
    "webdataset",
    "chump",
    "networkx==3.2.1",
    "roma",
    "joblib",
    "seaborn",
    "wandb",
    "appdirs",
    "jsonlines",
    "xtcocotools",
    "loguru",
    "optree",
    "fvcore",
    "pycocotools",
    "huggingface_hub",
    "numpy",
)

# Not oversights -- listed explicitly so validate_no_excluded_dependencies()
# (and the test suite) can assert none of them ever creep back into
# SAM3D_CORE_PIP_DEPENDENCIES for this minimal smoke test.
EXCLUDED_FROM_MINIMAL_PHASE_A: tuple[str, ...] = (
    "chumpy",  # wrong package name -- see module docstring
    "detectron2",  # no learned detector for this smoke test (Task 03C section 2/9)
    "moge",  # optional FOV estimator, not required (Task 03C section 3)
    "sam2",
    "sam-2",
    "sam3",
    "sam-3",  # optional segmentation, not required (Task 03C section 3)
)


class ExcludedDependencyError(AssertionError):
    pass


def validate_no_excluded_dependencies() -> None:
    """Raise ExcludedDependencyError if any excluded package name shows up
    in the core dependency list. Called by the notebook itself before
    installing anything, not only by tests -- a future edit that
    reintroduces one of these should fail immediately and loudly, not
    silently reappear as a wheel-build error three cells later."""
    lowered = [dep.lower() for dep in SAM3D_CORE_PIP_DEPENDENCIES]
    for excluded in EXCLUDED_FROM_MINIMAL_PHASE_A:
        for dep in lowered:
            if excluded in dep:
                raise ExcludedDependencyError(
                    f"'{excluded}' must not appear in SAM3D_CORE_PIP_DEPENDENCIES "
                    f"(found in {dep!r}) -- Task 03C excludes it explicitly from "
                    f"the minimal Phase A environment"
                )


def pip_install_command(python_executable: str) -> str:
    """Build the exact pip install command for the core dependency list,
    so the notebook's actual command and this module's tested data can
    never drift apart."""
    validate_no_excluded_dependencies()
    deps = " ".join(SAM3D_CORE_PIP_DEPENDENCIES)
    return f"{python_executable} -m pip install -q {deps}"


def torch_install_command(python_executable: str) -> str:
    """Build the exact pinned torch/torchvision install command."""
    pin = TORCH_PIN
    return (
        f"{python_executable} -m pip install -q "
        f"torch=={pin['torch']} torchvision=={pin['torchvision']} "
        f"--index-url {pin['index_url']}"
    )
