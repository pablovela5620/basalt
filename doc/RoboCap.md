# RoboCap offline input

This fork reads the native RoboCap layout with two camera sets.
`--dataset-type robocap` uses the four coverage cameras:

| Basalt camera | RoboCap device | Coverage |
|---|---|---|
| 0 | `dev4` | left |
| 1 | `dev1` | left-front |
| 2 | `dev5` | right-front |
| 3 | `dev3` | right |

`--dataset-type robocap-stereo` uses only the front pair (`dev1`, `dev5`) as
cameras 0 and 1. For a ready-made run on the example session, use
`pixi run robocap` or `pixi run robocap stereo`.

Both sets deliberately ignore the eye pair. The loader decodes H.264 to
grayscale and downscales the native 1920×1080 frames 2× to 960×540 during
decode (VIO throughput is bounded by feature detection, which scales with
pixel area; on a DGX Spark the downscale takes session 15 from 19 to 51 fps
with an aligned trajectory difference of 0.5% of path length). It groups each
set's hardware timestamps within 1 ms and uses the middle IMU (`dev0`).
Gyroscope samples define the output IMU clock; accelerometer samples are
linearly interpolated onto it. The loader adds the measured 14,902,432 ns
camera-to-IMU shift, so generated Basalt calibration files store
`cam_time_offset_ns = 0`. To keep the native resolution, construct `RobocapIO`
with `downscale = 1` and convert the calibration with `--downscale 1`.

Convert the factory Kalibr calibration; its intrinsics are scaled to the
reader's resolution (`--downscale`, default 2). One run writes both files: the
four-camera coverage calibration at `--output`, and the front-stereo one beside
it as `<stem>-stereo.json` unless `--stereo-output` overrides it:

```sh
pixi run robocap-calib -- \
  --factory-calibration /path/to/0factory-calibration-SERIAL \
  --output /path/to/robocap-basalt-calib.json
```

Run coverage VIO and save a Rerun recording (for the front pair, swap in
`robocap-stereo` and the `-stereo` calibration):

```sh
pixi run ./build/basalt_vio \
  --show-gui 0 \
  --dataset-type robocap \
  --dataset-path /path/to/SERIAL_session_N \
  --cam-calib /path/to/robocap-basalt-calib.json \
  --config-path data/msd/msdmo_config.json \
  --rerun-rrd /path/to/robocap.rrd
```

Run the adapter tests with `pixi run robocap-test`.
