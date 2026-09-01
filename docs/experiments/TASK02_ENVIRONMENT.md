# Task 02 — Execution Environment Probe

Recorded 2026-09-01, before installing any large dependencies, per Task 02
section 4. All values below were measured directly in this session, not
estimated.

| Property | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS (Noble Numbat), kernel `6.18.44-fc-v22` |
| Architecture | x86_64 |
| CPU | Intel(R) Xeon(R) Processor @ 2.10GHz |
| CPU cores | 4 (`nproc` = 4, all online) |
| RAM | 15 GiB total, ~15 GiB available at probe time |
| Swap | 0 B (none configured) |
| GPU | **None present.** `nvidia-smi`: not found. `/proc/driver/nvidia`: does not exist. `lspci \| grep -i vga`: no output. `dmesg`: only generic `vgaarb: loaded`, no NVIDIA/AMD driver load lines. |
| CUDA | Not available — `nvcc` not found, no driver, `torch.cuda.is_available()` untestable until torch install but expected `False` given no GPU device exists at all |
| Python | 3.11.15 at `/usr/local/bin/python3` |
| pip | 24.0 |
| Disk (`/`) | 252 GB total, 7.1 GB used, **30 GB available** at probe time (this is the effective ceiling for checkpoints + deps) |
| Other mounts | `/dev/shm` 16 GiB tmpfs; `/opt/claude-code` and `/opt/env-runner` are small system mounts, not usable for our data |
| Network (PyPI) | Working — test download of a 16.9 MB `numpy` wheel completed in ~1s |
| git-lfs | Not installed (`git lfs` unrecognized) |
| Pre-installed relevant packages | None of torch/numpy/pytorch3d/trimesh/opencv were present before this task |

## Implication for Task 02

**This environment has no GPU.** SAM 3D Body's smallest published checkpoint
is a 631M-parameter ViT-H DINOv3 backbone (per Task 01 research); the larger
variant is 840M parameters. Running a model of this size on 4 CPU cores with
no CUDA is expected to be dramatically slower than on any GPU — plausibly
minutes to tens of minutes per single image, if it runs at all within
available memory and without an unsupported-op failure (some vision
transformer implementations use fused/CUDA-only kernels that simply do not
have a CPU fallback).

Per the task's constraints (no purchasing, no automatic spend), this task
proceeds with a CPU-only attempt first, log exactly what happens (works
slowly / fails with an op-support error / fails on OOM / etc.), and treats
that outcome as primary compute evidence rather than a synthetic estimate.
Section "Compute Measurement" in
`TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md` records the actual result.

30 GB of free disk is enough for a single SAM 3D Body checkpoint (order of
a few GB) plus PyTorch and supporting libraries, but leaves little margin —
disk usage is tracked explicitly during install.
