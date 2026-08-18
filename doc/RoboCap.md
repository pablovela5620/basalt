# RoboCap offline input

This fork supports one native RoboCap layout through `--dataset-type robocap`:

| Basalt camera | RoboCap device | Coverage |
|---|---|---|
| 0 | `dev4` | left |
| 1 | `dev1` | left-front |
| 2 | `dev5` | right-front |
| 3 | `dev3` | right |

It deliberately ignores the eye pair. Images remain at 1920×1080. The loader
decodes H.264 to grayscale, groups the four hardware timestamps within 1 ms,
and uses the middle IMU (`dev0`). Gyroscope samples define the output IMU
clock; accelerometer samples are linearly interpolated onto it. The loader adds
the measured 14,902,432 ns camera-to-IMU shift, so generated Basalt calibration
files store `cam_time_offset_ns = 0`.

Convert the factory Kalibr calibration. One run writes the four-camera
coverage file at `--output` and the front-stereo pair beside it
(`robocap-basalt-calib-stereo.json`):

```sh
pixi run robocap-calib -- \
  --factory-calibration /path/to/0factory-calibration-SERIAL \
  --output /path/to/robocap-basalt-calib.json
```

Run four-camera VIO and save a Rerun recording (for the front stereo pair,
use `--dataset-type robocap-stereo` with the `-stereo` calibration):

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
