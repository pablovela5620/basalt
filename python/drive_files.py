"""Step-A validation driver: raw RoboCap files -> VIT tracker -> trajectory csv.

Mirrors src/io/dataset_io_robocap.cpp exactly — same coverage cameras, frameset
median timestamps, SWS_AREA downscale, and accel-onto-gyro interpolation — so a
trajectory mismatch against the file-fed basalt_vio golden run isolates to the
ctypes binding, not the data preparation. Throwaway by design; the production
driver feeds from the Rerun catalog instead of files.

Run from the repository root:
    pixi run -e driver python python/drive_files.py --session-dir /mnt/nas/datasets/robocap/f408193e6447b3b0_session_15
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import basalt_convert_robocap_calib as calib_converter  # noqa: E402
from vit_binding import Tracker  # noqa: E402

CAMERA_TO_IMU_OFFSET_NS = 14_902_432
FRAMESET_TOLERANCE_NS = 1_000_000
GYRO_SCALE = 0.000266316
ACCEL_SCALE = 0.001197101
COVERAGE_DEVICES = ((4, "left"), (1, "left-front"), (5, "right-front"), (3, "right"))


def index_video(path: Path) -> list[tuple[int, int]]:
    """(pts, timestamp_ns) per packet; timestamp = comment epoch + pts, no offset."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        comment = container.metadata.get("comment") or stream.metadata.get("comment")
        if comment is None:
            raise ValueError(f"Missing absolute timestamp comment in {path}")
        epoch_ns = int(comment) * 1_000
        time_base = stream.time_base
        frames = [
            (packet.pts, epoch_ns + int(packet.pts * time_base * Fraction(1_000_000_000)))
            for packet in container.demux(stream)
            if packet.pts is not None
        ]
    if not frames:
        raise ValueError(f"No timestamped video packets in {path}")
    return frames


def build_framesets(camera_frames: list[list[tuple[int, int]]]) -> list[tuple[int, list[int]]]:
    """Mirror the reader: anchor camera 0, nearest match per camera, median + offset.

    Returns (timestamp_ns_on_imu_clock, per-camera frame index) per complete frameset.
    """
    framesets: list[tuple[int, list[int]]] = []
    cursors = [0] * len(camera_frames)
    for anchor_index, (_, anchor_ns) in enumerate(camera_frames[0]):
        selected = [anchor_index]
        complete = True
        for camera in range(1, len(camera_frames)):
            frames = camera_frames[camera]
            index = cursors[camera]
            if index >= len(frames):
                complete = False
                break
            while index + 1 < len(frames) and abs(frames[index + 1][1] - anchor_ns) <= abs(frames[index][1] - anchor_ns):
                index += 1
            if abs(frames[index][1] - anchor_ns) > FRAMESET_TOLERANCE_NS:
                if frames[index][1] < anchor_ns:
                    cursors[camera] = index + 1
                complete = False
                break
            cursors[camera] = index
            selected.append(index)
        if not complete:
            continue
        times = sorted(camera_frames[camera][selected[camera]][1] for camera in range(len(camera_frames)))
        middle = len(times) // 2
        median = times[middle] if len(times) % 2 == 1 else times[middle - 1] + (times[middle] - times[middle - 1]) // 2
        framesets.append((median + CAMERA_TO_IMU_OFFSET_NS, selected))
    return framesets


def load_imu(session_dir: Path, session: int) -> np.ndarray:
    """Paired (t_ns, gyro_xyz, accel_xyz) rows: accel interpolated onto gyro times."""
    channels: dict[str, list[np.ndarray]] = {"gyro_data": [], "acc_data": []}
    for db_path in sorted(session_dir.glob(f"IMUWriter_dev0_session{session}_segment*.db")):
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as database:
            for table, parts in channels.items():
                rows = database.execute(f"SELECT x, y, z, timestamp FROM {table}").fetchall()
                parts.append(np.asarray(rows, dtype=np.int64))
    merged: dict[str, np.ndarray] = {}
    for table, parts in channels.items():
        raw = np.concatenate(parts)
        raw = raw[np.argsort(raw[:, 3], kind="stable")]
        keep = np.ones(len(raw), dtype=bool)
        keep[1:] = raw[1:, 3] != raw[:-1, 3]  # sort_and_deduplicate keeps the first
        merged[table] = raw[keep]
    gyro, accel = merged["gyro_data"], merged["acc_data"]

    paired: list[tuple[int, np.ndarray, np.ndarray]] = []
    accel_index = 0
    for t_ns, value in zip(gyro[:, 3], gyro[:, :3].astype(np.float64), strict=True):
        while accel_index + 1 < len(accel) and accel[accel_index + 1, 3] < t_ns:
            accel_index += 1
        if t_ns < accel[accel_index, 3] or accel_index + 1 >= len(accel):
            continue
        before, after = accel[accel_index], accel[accel_index + 1]
        interval = float(after[3] - before[3])
        alpha = 0.0 if interval == 0.0 else float(t_ns - before[3]) / interval
        interpolated = before[:3].astype(np.float64) + alpha * (after[:3] - before[:3]).astype(np.float64)
        paired.append((int(t_ns), value * GYRO_SCALE, interpolated * ACCEL_SCALE))
    print(f"{len(paired)} paired IMU samples")
    return paired


class SequentialDecoder:
    """Decode one camera's frames in frameset order (strictly increasing pts)."""

    def __init__(self, path: Path, downscale: int) -> None:
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.decoder = self.container.decode(self.stream)
        self.downscale = downscale
        self.reformatter = av.video.reformatter.VideoReformatter()
        self.current: av.VideoFrame | None = None

    def frame_at(self, pts: int) -> np.ndarray:
        while self.current is None or self.current.pts < pts:
            self.current = next(self.decoder)
        if self.current.pts != pts:
            raise ValueError(f"Decoder missed pts {pts} (at {self.current.pts})")
        target_width = max(self.current.width // self.downscale, 1)
        target_height = max(self.current.height // self.downscale, 1)
        gray = self.reformatter.reformat(
            self.current,
            width=target_width,
            height=target_height,
            format="gray8",
            interpolation="AREA" if self.downscale > 1 else "POINT",
        )
        return np.frombuffer(bytes(gray.planes[0]), dtype=np.uint8).reshape(target_height, gray.planes[0].line_size)[:, :target_width]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, default=Path("/mnt/nas/datasets/robocap/f408193e6447b3b0_session_15"))
    parser.add_argument("--factory-dir", type=Path, default=Path("/mnt/nas/datasets/robocap/0factory-calibration-f408193e6447b3b0"))
    parser.add_argument("--lib", type=Path, default=Path("build/libbasalt.so"))
    parser.add_argument("--config", type=Path, default=Path("python/robocap_vit.toml"))
    parser.add_argument("--downscale", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("datasets/vit_files_trajectory.csv"))
    args = parser.parse_args()

    session = int(args.session_dir.name.rsplit("_", 1)[1])
    calibration = calib_converter.convert(args.factory_dir, calib_converter.COVERAGE_CAMERA_SOURCES, args.downscale)["value0"]

    frames_per_camera: list[list[tuple[int, int]]] = []
    videos: list[list[Path]] = []
    for device, position in COVERAGE_DEVICES:
        paths = sorted(args.session_dir.glob(f"video_dev{device}_session{session}_segment*_{position}.mp4"))
        if not paths:
            raise FileNotFoundError(f"No videos for dev{device} {position}")
        videos.append(paths)
        indexed: list[tuple[int, int]] = []
        for path in paths:
            indexed.extend(index_video(path))
        indexed.sort(key=lambda frame: frame[1])
        frames_per_camera.append(indexed)
    framesets = build_framesets(frames_per_camera)
    print(f"{len(framesets)} complete framesets")
    imu = load_imu(args.session_dir, session)

    tracker = Tracker(args.lib, str(args.config), cam_count=len(COVERAGE_DEVICES))
    for index, (transform, intrinsics, resolution) in enumerate(
        zip(calibration["T_imu_cam"], calibration["intrinsics"], calibration["resolution"], strict=True)
    ):
        values = intrinsics["intrinsics"]
        quaternion = np.array([transform["qx"], transform["qy"], transform["qz"], transform["qw"]])
        x, y, z, w = quaternion
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        T_imu_cam = np.eye(4)
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

    decoders = [SequentialDecoder(paths[0], args.downscale) for paths in videos]
    if any(len(paths) > 1 for paths in videos):
        raise NotImplementedError("multi-segment sessions need per-segment decoder rollover")

    poses: list[tuple[int, float, float, float, float, float, float, float]] = []
    imu_cursor = 0
    started = time.monotonic()
    for count, (timestamp_ns, selected) in enumerate(framesets):
        # Lead the IMU past the frame time: in deterministic mode push_img blocks
        # until the backend emits this frame's state, and the backend needs an IMU
        # sample at/after the frame timestamp to integrate up to it. Feeding only
        # samples <= t deadlocks on the very first frameset.
        while imu_cursor < len(imu) and (imu_cursor == 0 or imu[imu_cursor - 1][0] <= timestamp_ns):
            t_ns, gyro_xyz, accel_xyz = imu[imu_cursor]
            tracker.push_imu(t_ns, gyro_xyz, accel_xyz)
            imu_cursor += 1
        for camera, frame_index in enumerate(selected):
            pts = frames_per_camera[camera][frame_index][0]
            image = np.ascontiguousarray(decoders[camera].frame_at(pts))
            tracker.push_img(camera, timestamp_ns, image.ctypes.data, width=image.shape[1], height=image.shape[0], stride=image.shape[1])
        poses.extend(tracker.poses())  # drain: deterministic mode fills a bounded queue
        if count % 100 == 0:
            print(f"frameset {count}/{len(framesets)}, {len(poses)} poses, {time.monotonic() - started:.1f}s", flush=True)
    tracker.stop()
    poses.extend(tracker.poses())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as sink:
        sink.write("#timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z\n")
        for t_ns, px, py, pz, qw, qx, qy, qz in poses:
            sink.write(f"{t_ns},{px},{py},{pz},{qw},{qx},{qy},{qz}\n")
    print(f"{len(poses)} poses -> {args.output} ({time.monotonic() - started:.1f}s)")
    tracker.close()


if __name__ == "__main__":
    main()
