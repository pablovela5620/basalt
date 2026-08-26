# VIT Python drivers

SLAM from Python through basalt's VIT C API (`libbasalt.so`). `drive_catalog.py`
feeds it straight from a Rerun catalog recording — the prototype for the
dataforge `slam` verb. `drive_files.py` feeds it from the raw RoboCap files,
mirroring `dataset_io_robocap.cpp` exactly, and exists only as the
binding-regression gate. `vit_binding.py` is the ctypes binding;
`robocap_feed.py` is the feed contract both drivers must share.

No running catalog server is needed: `--rrd` is a file path, and the driver
serves it through its own in-process `rr.server.Server`, querying it exactly
as it would the real catalog.

```bash
# SLAM one catalog recording (CUDA GPU required — NVDEC decode):
pixi run -e driver python python/drive_catalog.py --rrd <base.rrd> --output traj.csv

# Golden gates against the session-15 basalt_vio reference trajectory:
pixi run -e driver vit-golden                    # file-fed: binding fidelity, 5 cm
pixi run -e driver vit-golden-catalog <base.rrd> # catalog+NVDEC: decoder equivalence, 10 cm
```

Expected on session 15: 1588/1588 poses, ~0.13 cm ATE file-fed, ~6.5 cm
catalog-fed (NVDEC pixels differ from swscale gray8 by ~0.5 LSB, amplified by
deterministic VIO; frameset timestamps agree to 1 ns). The gotchas — IMU-lead
discipline, BT.601 luma, the `build-norerun/` arrow collision — are documented
where they live: the module docstrings and `pixi.toml`.
