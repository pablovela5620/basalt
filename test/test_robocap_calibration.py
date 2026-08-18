#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT: Path = Path(__file__).resolve().parents[1]
CONVERTER: Path = REPO_ROOT / "scripts" / "basalt_convert_robocap_calib.py"


def _camera_yaml(cameras: list[tuple[list[float], list[float], list[int], float]]) -> str:
    """Build a small Kalibr camera chain for a CLI integration test."""
    lines: list[str] = []
    for index, (translation_xyz, intrinsics, resolution, time_shift_s) in enumerate(cameras):
        lines.extend(
            [
                f"cam{index}:",
                "  T_cam_imu:",
                f"  - [1.0, 0.0, 0.0, {translation_xyz[0]}]",
                f"  - [0.0, 1.0, 0.0, {translation_xyz[1]}]",
                f"  - [0.0, 0.0, 1.0, {translation_xyz[2]}]",
                "  - [0.0, 0.0, 0.0, 1.0]",
                "  camera_model: pinhole",
                "  distortion_model: equidistant",
                "  distortion_coeffs: [0.1, 0.01, 0.001, 0.0001]",
                f"  intrinsics: {intrinsics}",
                f"  resolution: {resolution}",
                f"  timeshift_cam_imu: {time_shift_s}",
            ]
        )
    return "\n".join(lines) + "\n"


class RobocapCalibrationConverterTest(unittest.TestCase):
    def test_cli_writes_fixed_coverage_camera_order_and_imu_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root: Path = Path(temporary_directory)
            left_dir: Path = root / "imus_cam_l_extrinsic"
            front_dir: Path = root / "imus_cam_lr_front_extrinsic"
            right_dir: Path = root / "imus_cam_r_extrinsic"
            imu_dir: Path = root / "imus_intrinsic"
            for directory in (left_dir, front_dir, right_dir, imu_dir):
                directory.mkdir(parents=True)

            (left_dir / "imus_cam_l_extrinsic-camchain-imucam.yaml").write_text(
                _camera_yaml([([1.0, 0.0, 0.0], [101.0, 102.0, 103.0, 104.0], [1920, 1080], 0.019)]),
                encoding="utf-8",
            )
            (front_dir / "imus_cam_lr_front_extrinsic-camchain-imucam.yaml").write_text(
                _camera_yaml(
                    [
                        ([0.0, 2.0, 0.0], [201.0, 202.0, 203.0, 204.0], [1920, 1080], 0.012),
                        ([0.0, 0.0, 3.0], [301.0, 302.0, 303.0, 304.0], [1920, 1080], 0.013),
                    ]
                ),
                encoding="utf-8",
            )
            (right_dir / "imus_cam_r_extrinsic-camchain-imucam.yaml").write_text(
                _camera_yaml([([4.0, 0.0, 0.0], [401.0, 402.0, 403.0, 404.0], [1920, 1080], 0.017)]),
                encoding="utf-8",
            )
            (imu_dir / "imu_mid_0.yaml").write_text(
                "\n".join(
                    [
                        "accelerometer_noise_density: 0.006",
                        "accelerometer_random_walk: 0.0002",
                        "gyroscope_noise_density: 0.0007",
                        "gyroscope_random_walk: 0.00003",
                        "update_rate: 200.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            output_path: Path = root / "robocap-basalt-calib.json"
            result: subprocess.CompletedProcess[str] = subprocess.run(
                [
                    sys.executable,
                    str(CONVERTER),
                    "--factory-calibration",
                    str(root),
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            document: dict[str, object] = json.loads(output_path.read_text(encoding="utf-8"))
            calibration: dict[str, object] = document["value0"]  # type: ignore[assignment]

            # Default downscale 2 halves focal lengths and shifts the principal
            # point with the pixel-center convention: c' = (c + 0.5)/2 - 0.5.
            intrinsics: list[dict[str, object]] = calibration["intrinsics"]  # type: ignore[assignment]
            self.assertEqual([item["intrinsics"]["fx"] for item in intrinsics], [50.5, 100.5, 150.5, 200.5])  # type: ignore[index]
            self.assertEqual([item["intrinsics"]["fy"] for item in intrinsics], [51.0, 101.0, 151.0, 201.0])  # type: ignore[index]
            self.assertEqual([item["intrinsics"]["cx"] for item in intrinsics], [51.25, 101.25, 151.25, 201.25])  # type: ignore[index]
            self.assertEqual([item["intrinsics"]["cy"] for item in intrinsics], [51.75, 101.75, 151.75, 201.75])  # type: ignore[index]
            self.assertTrue(all(item["camera_type"] == "kb4" for item in intrinsics))

            transforms: list[dict[str, float]] = calibration["T_imu_cam"]  # type: ignore[assignment]
            self.assertEqual(
                [[item["px"], item["py"], item["pz"]] for item in transforms],
                [[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -3.0], [-4.0, 0.0, 0.0]],
            )
            self.assertEqual(calibration["resolution"], [[960, 540]] * 4)
            self.assertEqual(calibration["imu_update_rate"], 200.0)
            self.assertEqual(calibration["accel_noise_std"], [0.006] * 3)
            self.assertEqual(calibration["gyro_noise_std"], [0.0007] * 3)
            self.assertEqual(calibration["accel_bias_std"], [0.0002] * 3)
            self.assertEqual(calibration["gyro_bias_std"], [0.00003] * 3)
            self.assertEqual(calibration["cam_time_offset_ns"], 0)

            stereo_path: Path = root / "robocap-basalt-calib-stereo.json"
            self.assertTrue(stereo_path.is_file(), msg="stereo calibration must be written beside the coverage one")
            stereo_document: dict[str, object] = json.loads(stereo_path.read_text(encoding="utf-8"))
            stereo: dict[str, object] = stereo_document["value0"]  # type: ignore[assignment]

            stereo_intrinsics: list[dict[str, object]] = stereo["intrinsics"]  # type: ignore[assignment]
            self.assertEqual(
                [item["intrinsics"]["fx"] for item in stereo_intrinsics],  # type: ignore[index]
                [100.5, 150.5],
                msg="stereo order must be left-front then right-front",
            )

            stereo_transforms: list[dict[str, float]] = stereo["T_imu_cam"]  # type: ignore[assignment]
            self.assertEqual(
                [[item["px"], item["py"], item["pz"]] for item in stereo_transforms],
                [[0.0, -2.0, 0.0], [0.0, 0.0, -3.0]],
            )
            self.assertEqual(stereo["resolution"], [[960, 540]] * 2)
            self.assertEqual(stereo["imu_update_rate"], 200.0)

            unit_output: Path = root / "native.json"
            unit_result: subprocess.CompletedProcess[str] = subprocess.run(
                [
                    sys.executable,
                    str(CONVERTER),
                    "--factory-calibration",
                    str(root),
                    "--output",
                    str(unit_output),
                    "--downscale",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(unit_result.returncode, 0, msg=unit_result.stderr)
            native: dict[str, object] = json.loads(unit_output.read_text(encoding="utf-8"))["value0"]
            native_intrinsics: list[dict[str, object]] = native["intrinsics"]  # type: ignore[assignment]
            self.assertEqual([item["intrinsics"]["fx"] for item in native_intrinsics], [101.0, 201.0, 301.0, 401.0])  # type: ignore[index]
            self.assertEqual([item["intrinsics"]["cx"] for item in native_intrinsics], [103.0, 203.0, 303.0, 403.0])  # type: ignore[index]
            self.assertEqual(native["resolution"], [[1920, 1080]] * 4)


if __name__ == "__main__":
    unittest.main()
