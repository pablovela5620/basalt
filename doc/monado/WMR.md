# Windows Mixed Reality Headsets

Monado works out of the box with WMR devices and this Basalt: the WMR driver
reads the factory calibration from the headset and hands it to Basalt through
the VIT interface, so no config file is needed for a first run.

This fork builds everything with [pixi](https://pixi.prefix.dev). The overall
setup:

1. Build `libbasalt.so` here: `pixi run _build-all`, which fetches submodules,
   configures, and compiles every target (the `basalt` target implements VIT
   2.0, `thirdparty/vit/vit_interface.h`).
2. Build Monado (pixi-packaged mirror: `github.com/pablovela5620/monado`,
   branch `pixi`): `pixi run configure-then-build`.
3. Install udev rules so the headset is accessible without root.
4. Run:

```bash
cd ~/0Dev/repos/monado
VIT_SYSTEM_LIBRARY_PATH=$HOME/0Dev/repos/basalt/build/libbasalt.so \
pixi run ./build/src/xrt/targets/service/monado-service
```

Monado dlopens `libbasalt.so` at runtime — there is no build-time coupling
between the two projects.

## Useful environment variables

- `VIT_SYSTEM_LIBRARY_PATH`: absolute path to `libbasalt.so`.
- `SLAM_CONFIG`: path to a Basalt tracker `toml` (see `data/vit/*.toml.in`).
  Without it, sensor calibration comes from the WMR device's factory JSON.
- `SLAM_SUBMIT_FROM_START=true|false`: feed frames to Basalt immediately, or
  wait for the "Submit data to SLAM" checkbox in the debug GUI.
- `XRT_DEBUG_GUI=on`: Monado's own debug GUI (needs a display).
- `WMR_AUTOEXPOSURE=off` + the "WMR Camera" GUI box: manual exposure/gain.
- `WMR_LOG=debug`: prints the headset's JSON config block on start
  (`DEBUG [wmr_read_config] JSON config:`) if you want to inspect or convert
  the factory calibration.

## Live Rerun visualization

The VIT path can stream to [Rerun](https://rerun.io) (viewer 0.36):

- `BASALT_VIT_RERUN=spawn` — spawn a local viewer;
  `=<url>` — connect to a running viewer (`rerun+http://…/proxy`);
  `=<path>.rrd` — record to a file. Unset = disabled, zero overhead.
- `BASALT_VIT_RERUN_IMG_STRIDE=N` — log every Nth stereo frame (default 1).
- `BASALT_VIT_RERUN_IMU=1` — also log gyro/accel/bias streams.

Logged: stereo frames with tracked-feature overlays, camera calibration
(frusta), 6DoF pose + trajectory, velocity and IMU bias plots.

## Headset-independent smoke test

You can validate the whole Monado→Basalt pipeline without a headset using a
[Monado SLAM Dataset](https://huggingface.co/datasets/collabora/monado-slam-datasets)
sequence and the euroc driver:

```bash
# batch (headless; keep stdin open — slambatch exits on stdin EOF)
sleep 300 | VIT_SYSTEM_LIBRARY_PATH=…/libbasalt.so \
  ./build/src/xrt/targets/cli/monado-cli slambatch \
  <sequence_dir> <tracker.toml> <output_dir>

# or as a live SLAM-tracked HMD with the null compositor
XRT_COMPOSITOR_NULL=1 EUROC_HMD=true EUROC_PATH=<sequence_dir> \
VIT_SYSTEM_LIBRARY_PATH=…/libbasalt.so SLAM_CONFIG=<tracker.toml> \
SLAM_SUBMIT_FROM_START=true ./build/src/xrt/targets/service/monado-service
```

For MSD Odyssey+ (MOO) sequences use a `toml` pointing at
`data/msd/msdmo_calib.json` + `data/msd/msdmo_config.json` (see
`data/vit/msdmo.toml.in`; use absolute paths and `show-gui=0` when headless).

## Recalibrating my device

Roughly: print a Kalibr aprilgrid target, record a EuRoC-style dataset from
Monado moving the headset around the target, then run
[basalt_calibrate](https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/doc/Calibration.md#camera-calibration)
and
[basalt_calibrate_vio](https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/doc/Calibration.md#camera-imu-mocap-calibration)
on it.
