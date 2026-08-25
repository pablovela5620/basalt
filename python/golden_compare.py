"""Golden gate: compare two trajectory csvs by rigid-aligned ATE RMSE.

Both files: "#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z".
Exit code 0 iff ATE RMSE < --tolerance-m and pose counts agree within --count-slack.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from jaxtyping import Float64, Int64
from numpy import ndarray

ASSOCIATION_TOLERANCE_NS = 5_000_000


def load(path: Path) -> tuple[Int64[ndarray, "n"], Float64[ndarray, "n 3"]]:
    """Timestamps (ns) and positions from one trajectory csv."""
    rows: Float64[ndarray, "n 8"] = np.loadtxt(path, delimiter=",", skiprows=1)
    return rows[:, 0].astype(np.int64), rows[:, 1:4]


@dataclass
class Config:
    """Gate a candidate trajectory against the golden one."""

    golden: tyro.conf.Positional[Path]
    """Reference trajectory csv."""
    candidate: tyro.conf.Positional[Path]
    """Trajectory csv under test."""
    tolerance_m: float = 0.05
    """Maximum rigid-aligned ATE RMSE in metres."""
    count_slack: float = 0.02
    """Allowed relative pose-count difference."""


def main(args: Config) -> None:

    golden_t, golden_p = load(args.golden)
    candidate_t, candidate_p = load(args.candidate)
    count_ratio = abs(len(candidate_t) - len(golden_t)) / len(golden_t)

    indices = np.searchsorted(candidate_t, golden_t)
    indices = np.clip(indices, 1, len(candidate_t) - 1)
    indices -= np.abs(candidate_t[indices - 1] - golden_t) < np.abs(candidate_t[indices] - golden_t)
    matched = np.abs(candidate_t[indices] - golden_t) <= ASSOCIATION_TOLERANCE_NS
    source, target = candidate_p[indices[matched]], golden_p[matched]
    if matched.sum() < 10:
        sys.exit(f"FAIL: only {matched.sum()} associated poses")

    # Umeyama, rigid (no scale): align candidate onto golden.
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((target - target_mean).T @ (source - source_mean))
    sign = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vt
    aligned = (rotation @ (source - source_mean).T).T + target_mean
    errors = np.linalg.norm(aligned - target, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))

    print(f"poses: golden={len(golden_t)} candidate={len(candidate_t)} associated={int(matched.sum())} (count delta {count_ratio:.1%})")
    print(f"ATE  : rmse={rmse * 100:.2f} cm  max={errors.max() * 100:.2f} cm  median={np.median(errors) * 100:.2f} cm")
    if rmse >= args.tolerance_m or count_ratio > args.count_slack:
        sys.exit(f"FAIL: rmse {rmse:.4f} m (tolerance {args.tolerance_m}) or count delta {count_ratio:.1%} (slack {args.count_slack:.0%})")
    print("PASS")


if __name__ == "__main__":
    main(tyro.cli(Config))
