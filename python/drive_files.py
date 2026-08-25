"""Step-A validation driver: raw RoboCap files -> VIT tracker -> trajectory csv.

Mirrors src/io/dataset_io_robocap.cpp exactly — same coverage cameras, frameset
median timestamps, SWS_AREA downscale, and accel-onto-gyro interpolation — so a
trajectory mismatch against the file-fed basalt_vio golden run isolates to the
ctypes binding, not the data preparation. The production driver feeds from the
Rerun catalog instead of files; this one stays as the binding-fidelity gate.

Run from the repository root:
    pixi run -e driver python python/drive_files.py --session-dir /mnt/nas/datasets/robocap/f408193e6447b3b0_session_15
"""

from __future__ import annotations

import sqlite3
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import tyro
from jaxtyping import Bool, Float64, Int64, UInt8
from numpy import ndarray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import basalt_convert_robocap_calib as calib_converter  # noqa: E402
from robocap_feed import FrameStamp, Frameset, ImuSample, ImuSamples, build_framesets, feed_imu_lead  # noqa: E402
from vit_binding import PoseTuple, Tracker  # noqa: E402

GYRO_SCALE = 0.000266316
ACCEL_SCALE = 0.001197101
COVERAGE_DEVICES = ((4, "left"), (1, "left-front"), (5, "right-front"), (3, "right"))


def index_video(path: Path) -> list[FrameStamp]:
    """One FrameStamp per packet; timestamp = comment epoch + pts, no offset."""
    with av.open(str(path)) as container:
        stream: av.VideoStream = container.streams.video[0]
        comment: str | None = container.metadata.get("comment") or stream.metadata.get("comment")
        if comment is None:
            raise ValueError(f"Missing absolute timestamp comment in {path}")
        epoch_ns: int = int(comment) * 1_000
        time_base: Fraction = stream.time_base
        frames: list[FrameStamp] = [
            FrameStamp(packet.pts, epoch_ns + int(packet.pts * time_base * Fraction(1_000_000_000)))
            for packet in container.demux(stream)
            if packet.pts is not None
        ]
    if not frames:
        raise ValueError(f"No timestamped video packets in {path}")
    return frames


def load_imu(session_dir: Path, session: int) -> ImuSamples:
    """Paired (t_ns, gyro_xyz, accel_xyz) rows: accel interpolated onto gyro times."""
    channels: dict[str, list[Int64[ndarray, "rows 4"]]] = {"gyro_data": [], "acc_data": []}
    for db_path in sorted(session_dir.glob(f"IMUWriter_dev0_session{session}_segment*.db")):
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as database:
            for table, parts in channels.items():
                rows: list[tuple[int, int, int, int]] = database.execute(f"SELECT x, y, z, timestamp FROM {table}").fetchall()
                parts.append(np.asarray(rows, dtype=np.int64))
    merged: dict[str, Int64[ndarray, "rows 4"]] = {}
    for table, parts in channels.items():
        raw: Int64[ndarray, "rows 4"] = np.concatenate(parts)
        raw = raw[np.argsort(raw[:, 3], kind="stable")]
        keep: Bool[ndarray, "rows"] = np.ones(len(raw), dtype=bool)
        keep[1:] = raw[1:, 3] != raw[:-1, 3]  # sort_and_deduplicate keeps the first
        merged[table] = raw[keep]
    gyro: Int64[ndarray, "n_gyro 4"] = merged["gyro_data"]
    accel: Int64[ndarray, "n_accel 4"] = merged["acc_data"]

    paired: ImuSamples = []
    accel_index: int = 0
    for t_ns, value in zip(gyro[:, 3], gyro[:, :3].astype(np.float64), strict=True):
        while accel_index + 1 < len(accel) and accel[accel_index + 1, 3] < t_ns:
            accel_index += 1
        if t_ns < accel[accel_index, 3] or accel_index + 1 >= len(accel):
            continue
        before: Int64[ndarray, "4"] = accel[accel_index]
        after: Int64[ndarray, "4"] = accel[accel_index + 1]
        interval: float = float(after[3] - before[3])
        alpha: float = 0.0 if interval == 0.0 else float(t_ns - before[3]) / interval
        interpolated: Float64[ndarray, "3"] = before[:3].astype(np.float64) + alpha * (after[:3] - before[:3]).astype(np.float64)
        paired.append(ImuSample(int(t_ns), value * GYRO_SCALE, interpolated * ACCEL_SCALE))
    print(f"{len(paired)} paired IMU samples")
    return paired


class SequentialDecoder:
    """Decode one camera's frames in frameset order (strictly increasing pts)."""

    def __init__(self, path: Path, downscale: int) -> None:
        self.container: av.container.InputContainer = av.open(str(path))
        self.stream: av.VideoStream = self.container.streams.video[0]
        self.decoder: Iterator[av.VideoFrame] = self.container.decode(self.stream)
        self.downscale: int = downscale
        self.reformatter: av.video.reformatter.VideoReformatter = av.video.reformatter.VideoReformatter()
        self.current: av.VideoFrame | None = None

    def frame_at(self, pts: int) -> UInt8[ndarray, "h w"]:
        while self.current is None or self.current.pts < pts:
            self.current = next(self.decoder)
        if self.current.pts != pts:
            raise ValueError(f"Decoder missed pts {pts} (at {self.current.pts})")
        target_width: int = max(self.current.width // self.downscale, 1)
        target_height: int = max(self.current.height // self.downscale, 1)
        gray: av.VideoFrame = self.reformatter.reformat(
            self.current,
            width=target_width,
            height=target_height,
            format="gray8",
            interpolation="AREA" if self.downscale > 1 else "POINT",
        )
        return np.frombuffer(bytes(gray.planes[0]), dtype=np.uint8).reshape(target_height, gray.planes[0].line_size)[:, :target_width]


@dataclass
class Config:
    """Feed one raw RoboCap session through the VIT tracker, file-fed."""

    session_dir: Path = Path("datasets/robocap-example/f408193e6447b3b0_session_15")
    """Raw session directory (segment mp4s + IMU sqlite dbs)."""
    factory_dir: Path = Path("/mnt/nas/datasets/robocap/0factory-calibration-f408193e6447b3b0")
    """Raw Kalibr factory-calibration tree; not part of the HF sample."""
    lib: Path = Path("build/libbasalt.so")
    """libbasalt.so exporting the VIT C API."""
    vit_config: Path = Path("python/robocap_vit.toml")
    """VIT tracker config toml."""
    downscale: int = 3
    """Integer downscale applied to both frames and intrinsics."""
    output: Path = Path("datasets/vit_files_trajectory.csv")
    """Trajectory csv destination."""


def main(args: Config) -> None:
    session: int = int(args.session_dir.name.rsplit("_", 1)[1])
    calibration: dict = calib_converter.convert(args.factory_dir, calib_converter.COVERAGE_CAMERA_SOURCES, args.downscale)["value0"]

    frames_per_camera: list[list[FrameStamp]] = []
    videos: list[list[Path]] = []
    for device, position in COVERAGE_DEVICES:
        paths: list[Path] = sorted(args.session_dir.glob(f"video_dev{device}_session{session}_segment*_{position}.mp4"))
        if not paths:
            raise FileNotFoundError(f"No videos for dev{device} {position}")
        videos.append(paths)
        indexed: list[FrameStamp] = []
        for path in paths:
            indexed.extend(index_video(path))
        indexed.sort(key=lambda frame: frame.t_ns)
        frames_per_camera.append(indexed)
    framesets: list[Frameset] = build_framesets(frames_per_camera)
    print(f"{len(framesets)} complete framesets")
    imu: ImuSamples = load_imu(args.session_dir, session)

    tracker: Tracker = Tracker(args.lib, str(args.vit_config), cam_count=len(COVERAGE_DEVICES))
    for index, (transform, intrinsics, resolution) in enumerate(
        zip(calibration["T_imu_cam"], calibration["intrinsics"], calibration["resolution"], strict=True)
    ):
        values: dict = intrinsics["intrinsics"]
        quaternion: Float64[ndarray, "4"] = np.array([transform["qx"], transform["qy"], transform["qz"], transform["qw"]])
        x, y, z, w = quaternion
        rotation: Float64[ndarray, "3 3"] = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        T_imu_cam: Float64[ndarray, "4 4"] = np.eye(4)
        T_imu_cam[:3, :3] = rotation
        T_imu_cam[:3, 3] = [transform["px"], transform["py"], transform["pz"]]
        tracker.add_camera_calibration(
            index,
            width=resolution[0],
            height=resolution[1],
            frequency=30.0,
            fx=values["fx"],
            fy=values["fy"],
            cx=values["cx"],
            cy=values["cy"],
            distortion=[values["k1"], values["k2"], values["k3"], values["k4"]],
            T_imu_cam=T_imu_cam,
        )
    tracker.add_imu_calibration(
        frequency=calibration["imu_update_rate"],
        gyro_noise_std=calibration["gyro_noise_std"][0],
        gyro_bias_std=calibration["gyro_bias_std"][0],
        accel_noise_std=calibration["accel_noise_std"][0],
        accel_bias_std=calibration["accel_bias_std"][0],
    )
    tracker.start()

    decoders: list[SequentialDecoder] = [SequentialDecoder(paths[0], args.downscale) for paths in videos]
    if any(len(paths) > 1 for paths in videos):
        raise NotImplementedError("multi-segment sessions need per-segment decoder rollover")

    poses: list[PoseTuple] = []
    imu_cursor: int = 0
    started: float = time.monotonic()
    for count, (timestamp_ns, selected) in enumerate(framesets):
        imu_cursor = feed_imu_lead(tracker, imu, imu_cursor, timestamp_ns)
        for camera, frame_index in enumerate(selected):
            pts: int = frames_per_camera[camera][frame_index].opaque_id
            image: UInt8[ndarray, "h w"] = np.ascontiguousarray(decoders[camera].frame_at(pts))
            tracker.push_img(camera, timestamp_ns, image)
        poses.extend(tracker.drain_poses())  # deterministic mode fills a bounded queue
        if count % 100 == 0:
            print(f"frameset {count}/{len(framesets)}, {len(poses)} poses, {time.monotonic() - started:.1f}s", flush=True)
    tracker.stop()
    poses.extend(tracker.drain_poses())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as sink:
        sink.write("#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z\n")
        for t_ns, px, py, pz, qw, qx, qy, qz in poses:
            sink.write(f"{t_ns},{px},{py},{pz},{qw},{qx},{qy},{qz}\n")
    print(f"{len(poses)} poses -> {args.output} ({time.monotonic() - started:.1f}s)")
    tracker.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
