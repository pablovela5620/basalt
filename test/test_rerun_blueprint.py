"""Tests for the shared basalt_vio Rerun blueprint."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import rerun.blueprint as rrb

from scripts.rerun.basalt_vio_blueprint import build_blueprint, count_cameras, save_blueprint


def _walk(part: Any) -> list[Any]:
    """Return a blueprint part and all its descendants."""
    descendants: list[Any] = [part]
    for child in getattr(part, "contents", ()):
        if hasattr(child, "contents"):
            descendants.extend(_walk(child))
    return descendants


class BasaltVioBlueprintTest(unittest.TestCase):
    """Validate the intentional basalt_vio Viewer layout."""

    def test_blueprint_contains_four_cameras_world_and_sensor_plots(self) -> None:
        """The layout must expose every part of the four-camera VIO recording."""
        blueprint: rrb.Blueprint = build_blueprint(num_cameras=4)
        parts: list[Any] = _walk(blueprint.root_container)

        camera_views: list[Any] = [
            part
            for part in parts
            if getattr(part, "class_identifier", None) == "2D"
        ]
        self.assertEqual(
            [view.origin for view in camera_views],
            [f"/world/rig_0/cam_{index}/pinhole" for index in range(4)],
        )

        camera_grids: list[Any] = [
            part
            for part in parts
            if getattr(part, "name", None) == "Synchronized cameras"
        ]
        self.assertEqual(len(camera_grids), 1)
        self.assertEqual(camera_grids[0].grid_columns, 2)

        world_views: list[Any] = [
            part
            for part in parts
            if getattr(part, "class_identifier", None) == "3D"
        ]
        self.assertEqual(len(world_views), 1)
        self.assertEqual(world_views[0].origin, "/world")

        time_series_views: list[Any] = [
            part
            for part in parts
            if getattr(part, "class_identifier", None) == "TimeSeries"
        ]
        self.assertEqual(
            [view.name for view in time_series_views],
            ["Gyroscope", "Accelerometer", "VIO state"],
        )
        self.assertIn("/world/rig_0/imu_0/gyro", time_series_views[0].contents)
        self.assertIn("/world/rig_0/imu_0/accel", time_series_views[1].contents)

        self.assertTrue(blueprint.collapse_panels)
        self.assertFalse(blueprint.auto_views)
        self.assertFalse(blueprint.auto_layout)

    def test_blueprint_adapts_to_two_camera_rigs(self) -> None:
        """A stereo calibration must produce exactly two camera views."""
        blueprint: rrb.Blueprint = build_blueprint(num_cameras=2)
        parts: list[Any] = _walk(blueprint.root_container)

        camera_views: list[Any] = [
            part
            for part in parts
            if getattr(part, "class_identifier", None) == "2D"
        ]
        self.assertEqual(
            [view.origin for view in camera_views],
            [f"/world/rig_0/cam_{index}/pinhole" for index in range(2)],
        )

    def test_count_cameras_reads_basalt_calibration(self) -> None:
        """Camera count must come from the calibration's intrinsics list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            calibration_path: Path = Path(temp_dir) / "calib.json"
            calibration_path.write_text(
                json.dumps({"value0": {"intrinsics": [{}, {}]}}),
                encoding="utf-8",
            )
            self.assertEqual(count_cameras(calibration_path), 2)

    def test_blueprint_serializes_to_nonempty_rbl(self) -> None:
        """The generated blueprint must be loadable as a Rerun artifact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path: Path = Path(temp_dir) / "basalt_vio.rbl"
            save_blueprint(output_path, num_cameras=4)

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
