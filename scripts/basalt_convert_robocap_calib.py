#!/usr/bin/env python3

"""Convert RoboCap factory Kalibr files to a Basalt four-camera calibration."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro
import yaml


@dataclass(frozen=True, slots=True)
class CliArgs:
    """Command-line arguments for RoboCap calibration conversion."""

    factory_calibration: Path
    """Directory containing the unmodified RoboCap factory calibration."""
    output: Path
    """Destination Basalt cereal JSON file."""


@dataclass(frozen=True, slots=True)
class CameraSource:
    """One camera in the fixed coverage-camera order."""

    relative_path: Path
    """Kalibr camera-chain path relative to the factory directory."""
    camera_key: str
    """Camera key inside the Kalibr YAML document."""


CAMERA_SOURCES: tuple[CameraSource, ...] = (
    CameraSource(Path("imus_cam_l_extrinsic/imus_cam_l_extrinsic-camchain-imucam.yaml"), "cam0"),
    CameraSource(Path("imus_cam_lr_front_extrinsic/imus_cam_lr_front_extrinsic-camchain-imucam.yaml"), "cam0"),
    CameraSource(Path("imus_cam_lr_front_extrinsic/imus_cam_lr_front_extrinsic-camchain-imucam.yaml"), "cam1"),
    CameraSource(Path("imus_cam_r_extrinsic/imus_cam_r_extrinsic-camchain-imucam.yaml"), "cam0"),
)
IMU_RELATIVE_PATH: Path = Path("imus_intrinsic/imu_mid_0.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``.

    Args:
        path: YAML file to read.

    Returns:
        Parsed top-level mapping.
    """
    document: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return document


def _rotation_to_quaternion(rotation: list[list[float]]) -> tuple[float, float, float, float]:
    """Convert a 3×3 rotation matrix to an ``x, y, z, w`` quaternion."""
    trace: float = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        scale: float = math.sqrt(trace + 1.0) * 2.0
        quaternion: tuple[float, float, float, float] = (
            (rotation[2][1] - rotation[1][2]) / scale,
            (rotation[0][2] - rotation[2][0]) / scale,
            (rotation[1][0] - rotation[0][1]) / scale,
            0.25 * scale,
        )
    else:
        diagonal_index: int = max(range(3), key=lambda index: rotation[index][index])
        if diagonal_index == 0:
            scale = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
            quaternion = (
                0.25 * scale,
                (rotation[0][1] + rotation[1][0]) / scale,
                (rotation[0][2] + rotation[2][0]) / scale,
                (rotation[2][1] - rotation[1][2]) / scale,
            )
        elif diagonal_index == 1:
            scale = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
            quaternion = (
                (rotation[0][1] + rotation[1][0]) / scale,
                0.25 * scale,
                (rotation[1][2] + rotation[2][1]) / scale,
                (rotation[0][2] - rotation[2][0]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
            quaternion = (
                (rotation[0][2] + rotation[2][0]) / scale,
                (rotation[1][2] + rotation[2][1]) / scale,
                0.25 * scale,
                (rotation[1][0] - rotation[0][1]) / scale,
            )
    norm: float = math.sqrt(sum(component * component for component in quaternion))
    return tuple(component / norm for component in quaternion)  # type: ignore[return-value]


def _invert_kalibr_transform(matrix: list[list[float]]) -> dict[str, float]:
    """Invert Kalibr's camera-from-IMU transform for Basalt."""
    rotation_cam_imu: list[list[float]] = [[float(matrix[row][column]) for column in range(3)] for row in range(3)]
    rotation_imu_cam: list[list[float]] = [[rotation_cam_imu[column][row] for column in range(3)] for row in range(3)]
    translation_cam_imu: list[float] = [float(matrix[row][3]) for row in range(3)]
    translation_imu_cam: list[float] = [
        -sum(rotation_imu_cam[row][column] * translation_cam_imu[column] for column in range(3)) for row in range(3)
    ]
    quaternion_xyzw: tuple[float, float, float, float] = _rotation_to_quaternion(rotation_imu_cam)
    return {
        "px": translation_imu_cam[0],
        "py": translation_imu_cam[1],
        "pz": translation_imu_cam[2],
        "qx": quaternion_xyzw[0],
        "qy": quaternion_xyzw[1],
        "qz": quaternion_xyzw[2],
        "qw": quaternion_xyzw[3],
    }


def _camera_calibration(factory_calibration: Path, source: CameraSource) -> tuple[dict[str, float], dict[str, object], list[int]]:
    """Convert one fixed RoboCap camera from Kalibr to Basalt fields."""
    document: dict[str, Any] = _load_yaml(factory_calibration / source.relative_path)
    camera: dict[str, Any] = document[source.camera_key]
    matrix: list[list[float]] = camera["T_cam_imu"]
    intrinsics: list[float] = [float(value) for value in camera["intrinsics"]]
    distortion: list[float] = [float(value) for value in camera["distortion_coeffs"]]
    resolution: list[int] = [int(value) for value in camera["resolution"]]
    basalt_intrinsics: dict[str, object] = {
        "camera_type": "kb4",
        "intrinsics": {
            "fx": intrinsics[0],
            "fy": intrinsics[1],
            "cx": intrinsics[2],
            "cy": intrinsics[3],
            "k1": distortion[0],
            "k2": distortion[1],
            "k3": distortion[2],
            "k4": distortion[3],
        },
    }
    return _invert_kalibr_transform(matrix), basalt_intrinsics, resolution


def convert(factory_calibration: Path) -> dict[str, object]:
    """Convert a RoboCap factory directory into Basalt cereal JSON data."""
    transforms: list[dict[str, float]] = []
    intrinsics: list[dict[str, object]] = []
    resolutions: list[list[int]] = []
    for source in CAMERA_SOURCES:
        camera_result: tuple[dict[str, float], dict[str, object], list[int]] = _camera_calibration(factory_calibration, source)
        transforms.append(camera_result[0])
        intrinsics.append(camera_result[1])
        resolutions.append(camera_result[2])

    imu: dict[str, Any] = _load_yaml(factory_calibration / IMU_RELATIVE_PATH)
    accel_noise: float = float(imu["accelerometer_noise_density"])
    gyro_noise: float = float(imu["gyroscope_noise_density"])
    accel_walk: float = float(imu["accelerometer_random_walk"])
    gyro_walk: float = float(imu["gyroscope_random_walk"])
    calibration: dict[str, object] = {
        "T_imu_cam": transforms,
        "intrinsics": intrinsics,
        "resolution": resolutions,
        "vignette": [],
        "calib_accel_bias": [0.0] * 9,
        "calib_gyro_bias": [0.0] * 12,
        "imu_update_rate": float(imu["update_rate"]),
        "accel_noise_std": [accel_noise] * 3,
        "gyro_noise_std": [gyro_noise] * 3,
        "accel_bias_std": [accel_walk] * 3,
        "gyro_bias_std": [gyro_walk] * 3,
        "cam_time_offset_ns": 0,
    }
    return {"value0": calibration}


def main(args: CliArgs) -> None:
    """Write one converted Basalt calibration file."""
    document: dict[str, object] = convert(args.factory_calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=4) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main(tyro.cli(CliArgs))
