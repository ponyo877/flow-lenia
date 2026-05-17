"""Generate JAX convolution fixtures for the M1.6 L2 smoke test.

Re-generation (only needed if CASES is edited):

    python3 -m venv .venv-fixtures
    .venv-fixtures/bin/pip install -U pip
    .venv-fixtures/bin/pip install 'jax[cpu]' numpy
    .venv-fixtures/bin/python scripts/generate_jax_fixtures.py

Outputs are committed to git so CI does not need a JAX environment.

The fixtures pair:
  - the *exact* truncated kernel ``compute_kernel`` produces (size
    ``(2·er+1)²`` with ``er = ⌈(R+15)·r_i⌉``), normalised in f64 then cast
    to f32 — matching ``crate::kernel::compute_kernel``;
  - an activation drawn from a fixed-seed numpy ``Generator``;
  - the JAX ``jax.scipy.signal.convolve2d(..., mode='same',
    boundary='wrap'|'fill')`` result, cast to f32.

L2 tolerance is 1e-3 relative (DESIGN.md §4.5): the only source of
divergence with the Rust direct correlation is addition-order in the
summation/FFT path JAX takes internally. M6 bit-precision is out of scope.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp

CASES = [
    {
        "name": "case1_R5_r0.5_torus",
        "R": 5.0,
        "r": 0.5,
        "a": [0.25, 0.5, 0.75],
        "b": [1.0, 0.7, 0.4],
        "w": [0.05, 0.05, 0.05],
        "border": "torus",
        "grid_size": 32,
        "seed": 42,
    },
    {
        "name": "case2_R10_r0.7_torus",
        "R": 10.0,
        "r": 0.7,
        "a": [0.30, 0.60, 0.85],
        "b": [1.0, 0.5, 0.3],
        "w": [0.06, 0.06, 0.06],
        "border": "torus",
        "grid_size": 32,
        "seed": 1337,
    },
    {
        "name": "case3_R5_r0.5_wall",
        "R": 5.0,
        "r": 0.5,
        "a": [0.25, 0.5, 0.75],
        "b": [1.0, 0.7, 0.4],
        "w": [0.05, 0.05, 0.05],
        "border": "wall",
        "grid_size": 32,
        "seed": 42,
    },
]


def make_kernel(R: float, r: float, a, b, w) -> np.ndarray:
    """Mirror :rust:func:`crate::kernel::compute_kernel`.

    f64 accumulator + f64 normalisation, final cast to f32 — matches the
    Rust implementation's precision policy.
    """
    er = int(math.ceil((R + 15.0) * r))
    side = 2 * er + 1
    denom = (R + 15.0) * r

    K = np.zeros((side, side), dtype=np.float64)
    for y in range(-er, er + 1):
        for x in range(-er, er + 1):
            dist = math.sqrt(x * x + y * y) / denom
            # sigmoid(-(D-1)*10) where sigmoid(z) = 0.5*(tanh(z/2)+1)
            mask = 0.5 * (math.tanh(-(dist - 1.0) * 5.0) + 1.0)
            bump = sum(
                b[j] * math.exp(-((dist - a[j]) ** 2) / w[j]) for j in range(3)
            )
            K[y + er, x + er] = mask * bump
    K = K / K.sum()
    return K.astype(np.float32)


def make_activation(grid_size: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(size=(grid_size, grid_size)).astype(np.float32)


def write_f32_bin(path: Path, array: np.ndarray) -> None:
    array.astype(np.float32).tofile(path)


def jax_convolve(A: np.ndarray, K: np.ndarray, border: str) -> np.ndarray:
    """JAX-side reference convolution.

    ``jax.scipy.signal.convolve2d`` only supports ``boundary='fill',
    fillvalue=0`` (as of jax 0.10.0 — see :py:func:`jax.scipy.signal.convolve2d`
    NotImplementedError for the ``wrap`` branch). To get torus semantics we
    explicitly pad ``A`` with ``mode='wrap'`` and call ``mode='valid'`` so the
    wrapped neighbourhood is already inside the array; ``boundary='fill'``
    then doesn't trim anything because the padding already supplied every
    in-kernel cell.
    """
    kh, kw = K.shape
    er_y, er_x = kh // 2, kw // 2
    if border == "torus":
        A_padded = jnp.pad(jnp.asarray(A), ((er_y, er_y), (er_x, er_x)), mode="wrap")
        out = jsp.signal.convolve2d(
            A_padded, jnp.asarray(K), mode="valid", boundary="fill"
        )
    elif border == "wall":
        out = jsp.signal.convolve2d(
            jnp.asarray(A), jnp.asarray(K), mode="same", boundary="fill"
        )
    else:
        raise ValueError(f"unknown border {border!r}")
    return np.asarray(out, dtype=np.float32)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for case in CASES:
        K = make_kernel(case["R"], case["r"], case["a"], case["b"], case["w"])
        A = make_activation(case["grid_size"], case["seed"])
        out = jax_convolve(A, K, case["border"])

        name = case["name"]
        write_f32_bin(out_dir / f"{name}_A.bin", A)
        write_f32_bin(out_dir / f"{name}_K.bin", K)
        write_f32_bin(out_dir / f"{name}_out.bin", out)

        manifest.append(
            {
                "name": name,
                "grid_size": case["grid_size"],
                "kernel_size": K.shape[0],
                "border": case["border"],
                "R": case["R"],
                "r": case["r"],
                "seed": case["seed"],
            }
        )
        print(
            f"  {name}: grid={case['grid_size']}², "
            f"K={K.shape[0]}², border={case['border']}"
        )

    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(manifest)} fixtures + manifest.json to {out_dir}")


if __name__ == "__main__":
    main()
