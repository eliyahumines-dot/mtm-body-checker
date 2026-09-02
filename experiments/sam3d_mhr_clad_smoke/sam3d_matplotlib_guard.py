"""Headless-Matplotlib-backend guard for standalone Environment A worker/
check scripts -- Task 03F.

## Root cause

A real Colab run reached SAM3D_SOURCE_IMPORT (Task 03E's source-root
resolution was correct: the standalone import-check subprocess found and
inserted the right `sys.path` entry) and then failed *inside*
`import sam_3d_body` itself:

    ValueError: Key backend: 'module://matplotlib_inline.backend_inline'
    is not a valid value for backend

Colab's interactive IPython kernel sets `MPLBACKEND` (or configures
Matplotlib directly) to `module://matplotlib_inline.backend_inline` --
IPython's own inline-plot-display backend, registered only inside a live
IPython/Jupyter kernel process. A `subprocess.run(...)` launched from that
kernel inherits the *same* `os.environ` by default (Python does this
unless an explicit `env=` is given), including that `MPLBACKEND` value --
even though the child is a plain, non-interactive Environment A worker
process that never runs inside any IPython kernel and has no
`matplotlib_inline` package installed to satisfy it. Whenever SAM 3D Body
or one of its dependencies (e.g. `pyrender`) imports `matplotlib.pyplot`
during `import sam_3d_body`, Matplotlib tries to activate that inherited,
invalid-here backend name and raises.

## Why not just install `matplotlib-inline`

That would only mask this one specific inherited value, would still leave
a headless subprocess depending on an interactive-kernel-only package for
no functional reason, and would not fix the underlying problem: a
standalone worker process must never depend on whatever backend its
launching kernel happened to have configured. The fix here is to give
every standalone Environment A process an explicit, headless-safe backend
of its own, unconditionally -- not to satisfy the inherited one.
"""

from __future__ import annotations

import os
import sys

HEADLESS_BACKEND = "Agg"


def force_headless_matplotlib_backend() -> str | None:
    """Force ``MPLBACKEND`` to :data:`HEADLESS_BACKEND` in the CURRENT
    process's environment, returning whatever value was inherited
    beforehand (``None`` if it was unset).

    Must be called before importing matplotlib or anything that might
    import it as a side effect (``sam_3d_body``, ``pyrender``, etc.) --
    callers should invoke this as close to the top of the entry-point
    script as possible, before any such import.

    Deliberately assigns rather than ``setdefault()``: an inherited value
    (e.g. Colab's own ``module://matplotlib_inline.backend_inline``)
    already exists in a subprocess's environment by default and must be
    overridden, not preserved -- ``setdefault()`` would leave it in place.
    """
    inherited = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = HEADLESS_BACKEND
    return inherited


def sanitized_subprocess_env(base_env: dict | None = None) -> dict:
    """Build a ``subprocess.run(..., env=...)`` mapping with ``MPLBACKEND``
    forced to :data:`HEADLESS_BACKEND`, preserving every other variable
    from ``base_env`` untouched (defaults to a copy of the current
    process's ``os.environ`` if not given).

    For the NOTEBOOK's use when launching a standalone worker/check
    subprocess -- belt-and-suspenders alongside that subprocess's own
    in-process :func:`force_headless_matplotlib_backend` call, the same
    redundant-but-independent verification pattern Task 03D/03E already
    established for the source-root mechanism.
    """
    env = dict(base_env if base_env is not None else os.environ)
    env["MPLBACKEND"] = HEADLESS_BACKEND
    return env


def effective_matplotlib_backend() -> str | None:
    """Return Matplotlib's currently effective backend if matplotlib has
    already been imported (by this process directly, or as a side effect
    of importing something else), else ``None``.

    Deliberately never imports matplotlib itself just to check -- a
    project/test environment with no matplotlib installed at all (or
    where `import sam_3d_body` never got far enough to import it) must
    not fail this check or force an otherwise-unnecessary import.
    """
    mpl = sys.modules.get("matplotlib")
    if mpl is None:
        return None
    try:
        return mpl.get_backend()
    except Exception:
        return None


SAM3D_MATPLOTLIB_BACKEND_FAILURE = "SAM3D_MATPLOTLIB_BACKEND_FAILURE"
SAM3D_IMPORT_RUNTIME_DEPENDENCY_FAILURE = "SAM3D_IMPORT_RUNTIME_DEPENDENCY_FAILURE"

_BACKEND_ERROR_MARKERS = ("matplotlib", "backend")


def classify_import_exception(exc: BaseException) -> str:
    """Return a failure-category tag for an exception raised while
    importing ``sam_3d_body`` or one of its submodules, distinguishing the
    specific, now-understood Matplotlib-backend failure mode from a
    general runtime/dependency failure (Task 03F section 6: once the
    source root itself has already been validated, a subsequent import
    failure must not be mislabeled as a source-root/path problem).

    Returns one of :data:`SAM3D_MATPLOTLIB_BACKEND_FAILURE` (the exact
    class of error this task fixes) or :data:`SAM3D_IMPORT_RUNTIME_DEPENDENCY_FAILURE`
    (any other post-root-validation import-time exception). Never raises;
    stateless -- the caller is responsible for also recording
    ``str(exc)`` alongside this tag so the underlying exception is never
    discarded.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if all(marker in text for marker in _BACKEND_ERROR_MARKERS):
        return SAM3D_MATPLOTLIB_BACKEND_FAILURE
    return SAM3D_IMPORT_RUNTIME_DEPENDENCY_FAILURE
