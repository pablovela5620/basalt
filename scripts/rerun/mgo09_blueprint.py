"""Generate the default Rerun layout for the four-camera MGO09 recording."""

from __future__ import annotations

from pathlib import Path

import rerun.blueprint as rrb


APPLICATION_ID: str = "basalt_vio"
REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH: Path = (
    REPOSITORY_ROOT / "datasets/MGO09_short_1_updown/basalt_vio_mgo09.rbl"
)


def _camera_view(camera_index: int) -> rrb.Spatial2DView:
    """Create one synchronized camera view.

    Args:
        camera_index: Zero-based camera number in the Reverb G2 rig.

    Returns:
        A 2D view rooted at the camera's pinhole projection.
    """
    origin: str = f"/world/rig_0/cam_{camera_index}/pinhole"
    return rrb.Spatial2DView(
        name=f"Camera {camera_index}",
        origin=origin,
        contents=f"{origin}/**",
    )


def build_blueprint() -> rrb.Blueprint:
    """Build a compact layout for inspecting four-camera VIO.

    Returns:
        The Rerun blueprint shared by interactive and validation runs.
    """
    camera_grid: rrb.Grid = rrb.Grid(
        *[_camera_view(camera_index) for camera_index in range(4)],
        grid_columns=2,
        name="Four synchronized cameras",
    )

    world_view: rrb.Spatial3DView = rrb.Spatial3DView(
        name="Trajectory and landmarks",
        origin="/world",
        contents=[
            "/world/rig_0",
            "/world/rig_0/cam_0",
            "/world/rig_0/cam_0/pinhole",
            "/world/rig_0/cam_1",
            "/world/rig_0/cam_1/pinhole",
            "/world/rig_0/cam_2",
            "/world/rig_0/cam_2/pinhole",
            "/world/rig_0/cam_3",
            "/world/rig_0/cam_3/pinhole",
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
        name="Basalt four-camera VIO",
    )

    return rrb.Blueprint(
        layout,
        auto_layout=False,
        auto_views=False,
        collapse_panels=True,
    )


def save_blueprint(output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Write the MGO09 blueprint to disk.

    Args:
        output_path: Destination `.rbl` path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_blueprint().save(APPLICATION_ID, output_path)


def main() -> None:
    """Generate the default blueprint beside the MGO09 recording."""
    save_blueprint()
    print(DEFAULT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
