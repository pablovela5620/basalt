# Basalt for Monado

This is a fork of [Basalt](https://gitlab.com/VladyslavUsenko/basalt) improved
for tracking XR devices with
[Monado](https://gitlab.freedesktop.org/monado/monado). Many thanks to the
Basalt authors.

## This fork — pixi packaging & upstream mirroring

This repository (`pablovela5620/basalt`) is a **packaging fork** of the Monado
Basalt fork ([`mateosss/basalt`](https://gitlab.freedesktop.org/mateosss/basalt)).
It adds a one-command [pixi](https://pixi.sh) build and an optional
[rerun.io](https://rerun.io) visualization backend, while staying faithful to
upstream. There are two branches:

- **`main`** — a pristine mirror of upstream `mateosss/basalt`. Never committed
  to directly; only updated via `git pull upstream main`.
- **`pixi`** *(default)* — the branch you build: `main` **plus** additive pixi
  packaging (`pixi.toml`/`pixi.lock`, `.gitignore`) and the rerun visualization
  backend (opt-in at runtime with `--rerun`; compile-gated by
  `BASALT_ENABLE_RERUN`).

**Keeping it mirrored with upstream:**

```bash
git remote add upstream https://gitlab.freedesktop.org/mateosss/basalt.git  # once
git checkout main && git pull upstream main        # refresh the mirror
git checkout pixi && git rebase main               # replay packaging + rerun viz
git push --force-with-lease origin pixi
```

Rebases onto a newer upstream stay low-conflict because the SLAM core is
untouched. Rerun is a separate `basalt_rerun` library: its offline and live VIT
adapters own configuration, timelines, conversion, and logging. Upstream files
retain only small `#ifdef BASALT_RERUN` lifecycle hooks. Building with
`-DBASALT_ENABLE_RERUN=OFF` removes those hooks and the Rerun CLI options.

### Pixi quickstart

Install [Pixi](https://pixi.sh/latest/installation/) first. On macOS, also
install the Xcode Command Line Tools with `xcode-select --install`.

From a new terminal:

```bash
git clone --branch pixi https://github.com/pablovela5620/basalt.git
cd basalt
pixi run robocap               # everything: build + download + VIO + Rerun viewer
```

`pixi run robocap` fetches submodules, configures, compiles `basalt_vio`,
downloads the [RoboCap example session](https://huggingface.co/datasets/pablovela5620/robocap-example)
(153 MB into `datasets/robocap-example/`), runs four-camera VIO on it, and opens
the recording in the Rerun viewer. Pixi supplies the compiler, CMake, native
libraries, Python tools, FFmpeg, and Rerun — no environment activation needed.

The other visible tasks (`pixi task list` shows them; `_`-prefixed helpers are
chained intermediaries):

```bash
pixi run robocap stereo        # the same demo using only the front stereo pair
pixi run build                 # fetch submodules, configure, compile basalt_vio
pixi run msd                   # same demo on a Monado SLAM dataset (default MGO09)
pixi run msd MOO09_short_1_updown MO_odyssey_plus MOO_others   # any MSD sequence
pixi run robocap-calib         # RoboCap factory calibration -> Basalt JSON
pixi run robocap-test          # calibration + dataset reader + blueprint tests
```

`msd` takes three arguments — dataset name, device group, and subgroup — matching
the folder layout of [monado-slam-datasets](https://huggingface.co/datasets/collabora/monado-slam-datasets).
Both demos write `basalt_vio.rrd`, a TUM trajectory, and a shared Rerun layout
(`scripts/rerun/basalt_vio_blueprint.py`) under `datasets/`. For native RoboCap
data, follow the [RoboCap offline input guide](doc/RoboCap.md); it includes
calibration, four-camera VIO, Rerun recording, and adapter test commands.

## Installation

- **Prebuilt (Ubuntu/Raspberry/Radxa)**: Download [latest .deb](https://gitlab.freedesktop.org/mateosss/basalt/-/releases) and install with

  ```bash
  sudo apt install -y ./basalt-monado-*.deb
  ```

- **From source (Linux)**

  ```bash
  git clone --recursive https://gitlab.freedesktop.org/mateosss/basalt.git
  cd basalt && ./scripts/install_deps.sh
  cmake --preset library # use "development" instead of "library" if you want extra binaries and debug symbols
  sudo cmake --build build --target install
  ```

- **From source (Windows)**: See the [build guide](doc/monado/Windows.md) for Windows.

## Usage

If you want to run OpenXR application with Monado, you need to set the
environment variable `VIT_SYSTEM_LIBRARY_PATH` to the path of the basalt library.

By default, Monado will try to load the library from `/usr/lib/libbasalt.so` if
the environment variable is not set.

If you want to test whether everything is working you can download a short dataset with [EuRoC (ASL) format](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) format like [`MOO09_short_1_updown`](https://huggingface.co/datasets/collabora/monado-slam-datasets/resolve/main/M_monado_datasets/MO_odyssey_plus/MOO_others/MOO09_short_1_updown.zip?download=true) from the [Monado SLAM datasets](https://huggingface.co/datasets/collabora/monado-slam-datasets):

```bash
wget https://huggingface.co/datasets/collabora/monado-slam-datasets/resolve/main/M_monado_datasets/MO_odyssey_plus/MOO_others/MOO09_short_1_updown.zip
unzip MOO09_short_1_updown.zip
```

- **Try it standalone with a dataset (requires extra binaries)**

  ```bash
  basalt_vio --show-gui 1 --dataset-path MOO09_short_1_updown/ --dataset-type euroc --cam-calib /usr/share/basalt/msdmo_calib.json --config-path /usr/share/basalt/msdmo_config.json
  ```

- **Use a RealSense camera without Monado (requires extra binaries)**
  You'll need to calibrate your camera if you want the best results but meanwhile you can try with these calibration files instead.

  - RealSense D455 (and maybe also D435)

    ```bash
    basalt_rs_t265_vio --is-d455 --cam-calib /usr/share/basalt/d455_calib.json --config-path /usr/share/basalt/default_config.json
    ```

  - Realsense T265: Get t265_calib.json from [this issue](https://gitlab.com/VladyslavUsenko/basalt/-/issues/52) and run

    ```bash
    basalt_rs_t265_vio --cam-calib t265_calib.json --config-path /usr/share/basalt/default_config.json
    ```

- **Try it through `monado-cli` with a dataset**

  ```bash
  monado-cli slambatch MOO09_short_1_updown/ /usr/share/basalt/msdmo.toml results
  ```

- **Try it with `monado`, a dataset, and an OpenXR app**

  ```bash
  # Run monado-service with a fake "euroc device" driver
  export EUROC_PATH=MOO09_short_1_updown/ # dataset path
  export EUROC_HMD=false # false for controller tracking
  export EUROC_PLAY_FROM_START=true # produce samples right away
  export SLAM_CONFIG=/usr/share/basalt/msdmo.toml # includes calibration
  export SLAM_SUBMIT_FROM_START=true # consume samples right away
  export XRT_DEBUG_GUI=1 # enable monado debug ui
  monado-service &

  # Get and run a sample OpenXR application
  wget https://gitlab.freedesktop.org/wallbraker/apps/-/raw/main/VirtualGround-x86_64.AppImage
  chmod +x VirtualGround-x86_64.AppImage
  ./VirtualGround-x86_64.AppImage normal
  ```

- **Use a real device in Monado**.

  When using a real device driver you might want to enable the `XRT_DEBUG_GUI=1` and `SLAM_UI=1` environment variables to show debug GUIs of Monado and Basalt respectively.

  Monado has a couple of drivers supporting SLAM tracking (and thus Basalt). Most of them should work without any user input.

  - WMR ([troubleshoot](doc/monado/WMR.md))
  - Rift S (might need to press "Submit to SLAM", like the Vive Driver).
  - Northstar / DepthAI ([This hand-tracking guide](https://monado.freedesktop.org/handtracking) has a depthai section).
  - Vive Driver (Valve Index) ([read before using](doc/monado/Vive.md))
  - RealSense Driver ([setup](doc/monado/Realsense.md)).

## Development

If you want to set up your build environment for developing and iterating on Basalt, see the [development guide](doc/Development.md).
