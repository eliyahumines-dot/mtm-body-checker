"""Worker subprocess: load an MHR params JSON via clad-body and measure it.

Run in a *separate process* from run.py deliberately. Investigation (see
docs/experiments/TASK02_SAM3D_MHR_CLAD_SMOKE_TEST.md) found that
clad-body's MHR loader crashes the whole Python process (native segfault
inside pymomentum's FBX loader) in this environment. Isolating it in a
subprocess means that crash produces a clean non-zero/negative return code
that run.py can catch and report as a status, instead of taking down the
whole CLI. This mirrors clad-body's own internal pattern of shelling out to
a subprocess for the same import-order/isolation reason.

Usage:
    python _mhr_measure_worker.py <params_json_path> <output_json_path> [known_height_cm]

Writes a JSON result to <output_json_path> on success. On failure, prints
to stderr and exits non-zero (or the OS kills it with a signal, e.g. -11
for SIGSEGV, which the parent process's subprocess.run() reports as a
negative returncode).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rescale import compute_mesh_height_cm, uniform_rescale_to_height  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: _mhr_measure_worker.py <params_json> <output_json> [known_height_cm]", file=sys.stderr)
        return 2

    params_path = sys.argv[1]
    output_path = sys.argv[2]
    known_height_cm = float(sys.argv[3]) if len(sys.argv) > 3 else None

    from clad_body.load import load_mhr_from_params
    from clad_body.measure import measure

    body = load_mhr_from_params(params_path)

    import numpy as np

    raw_height_cm = compute_mesh_height_cm(np.asarray(body.mesh.vertices))

    rescale_factor = None
    if known_height_cm is not None:
        rescaled_verts, rescale_factor = uniform_rescale_to_height(
            np.asarray(body.mesh.vertices), known_height_cm
        )
        body.mesh.vertices = rescaled_verts

    measurements = measure(body)

    result = {
        "raw_body_height_cm": raw_height_cm,
        "known_height_cm": known_height_cm,
        "rescale_applied": known_height_cm is not None,
        "rescale_factor": rescale_factor,
        "measurements_cm": {k: (float(v) if not isinstance(v, str) else v) for k, v in measurements.items()},
        "n_vertices": body.n_vertices,
        "n_faces": body.n_faces,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
