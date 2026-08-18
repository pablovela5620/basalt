"""Generate the default Rerun layout for a basalt_vio recording.

The entity paths below are basalt_vio's own logging schema, so the layout
applies to any dataset the runner supports (Monado SLAM, RoboCap, ...).
The camera count is read from the Basalt calibration that produced the
recording, so 2-camera stereo rigs and 4-camera rigs share this script.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rerun.blueprint as rrb
import tyro


APPLICATION_ID: str = "basalt_vio"


@dataclass(frozen=True, slots=True)
class CliArgs:
    """Command-line arguments for blueprint generation."""

    output: Path
    """Destination `.rbl` path."""
    calibration: Path
    """Basalt calibration JSON of the run; sets the camera view count."""


def count_cameras(calibration_path: Path) -> int:
    """Read the camera count from a Basalt cereal calibration file.

    Args:
        calibration_path: Basalt calibration JSON (``{"value0": {...}}``).

    Returns:
        Number of cameras in the calibration.
    """
    document: dict[str, Any] = json.loads(calibration_path.read_text(encoding="utf-8"))
    cameras: int = len(document["value0"]["intrinsics"])
    if cameras < 1:
        raise ValueError(f"Calibration {calibration_path} lists no cameras")
    return cameras


def _camera_view(camera_index: int) -> rrb.Spatial2DView:
    """Create one synchronized camera view.

    Args:
        camera_index: Zero-based camera number in the rig.

    Returns:
        A 2D view rooted at the camera's pinhole projection.
    """
    origin: str = f"/world/rig_0/cam_{camera_index}/pinhole"
    return rrb.Spatial2DView(
        name=f"Camera {camera_index}",
        origin=origin,
        contents=f"{origin}/**",
    )


def build_blueprint(num_cameras: int) -> rrb.Blueprint:
    """Build a compact layout for inspecting multi-camera VIO.

    Args:
        num_cameras: Number of synchronized cameras in the rig.

    Returns:
        The Rerun blueprint shared by interactive and validation runs.
    """
    camera_grid: rrb.Grid = rrb.Grid(
        *[_camera_view(camera_index) for camera_index in range(num_cameras)],
        grid_columns=2,
        name="Synchronized cameras",
    )

    camera_contents: list[str] = [
        path
        for camera_index in range(num_cameras)
        for path in (f"/world/rig_0/cam_{camera_index}", f"/world/rig_0/cam_{camera_index}/pinhole")
    ]
    world_view: rrb.Spatial3DView = rrb.Spatial3DView(
        name="Trajectory and landmarks",
        origin="/world",
        contents=[
            "/world/rig_0",
            *camera_contents,
            "/world/landmarks",
            "/world/runs/basalt/trajectory",
            "/world/rig_0_path",
            "/world/rig_0_path/endpoints",
        ],
        line_grid=True,
    )

    gyro_view: rrb.TimeSeriesView = rrb.TimeSeriesView(
        name="Gyroscope",
        origin="/world/rig_0/imu_0",
        contents=["/world/rig_0/imu_0/gyro"],
        plot_legend=rrb.PlotLegend(visible=True),
    )
    accel_view: rrb.TimeSeriesView = rrb.TimeSeriesView(
        name="Accelerometer",
        origin="/world/rig_0/imu_0",
        contents=["/world/rig_0/imu_0/accel"],
        plot_legend=rrb.PlotLegend(visible=True),
    )
    state_view: rrb.TimeSeriesView = rrb.TimeSeriesView(
        name="VIO state",
        origin="/world",
        contents=[
            "/world/metrics/velocity",
            "/world/rig_0/imu_0/bias_gyro",
            "/world/rig_0/imu_0/bias_accel",
        ],
        plot_legend=rrb.PlotLegend(visible=True),
    )

    layout: rrb.Vertical = rrb.Vertical(
        rrb.Horizontal(
            camera_grid,
            world_view,
            column_shares=[2.0, 1.0],
            name="Vision and reconstruction",
        ),
        rrb.Horizontal(
            gyro_view,
            accel_view,
            state_view,
            column_shares=[1.0, 1.0, 1.0],
            name="Sensors and state",
        ),
        row_shares=[3.0, 1.0],
        name="Basalt VIO",
    )

    return rrb.Blueprint(
        layout,
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def save_blueprint(output_path: Path, num_cameras: int) -> None:
    """Write the blueprint to disk.

    Args:
        output_path: Destination `.rbl` path.
        num_cameras: Number of synchronized cameras in the rig.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_blueprint(num_cameras).save(APPLICATION_ID, output_path)


def main(args: CliArgs) -> None:
    """Generate the blueprint sized to the run's calibration."""
    save_blueprint(args.output, count_cameras(args.calibration))
    print(args.output)


if __name__ == "__main__":
    main(tyro.cli(CliArgs))
