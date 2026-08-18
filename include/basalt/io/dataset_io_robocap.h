/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2026, Pablo Vela.
All rights reserved.
*/

#pragma once

#include <basalt/io/dataset_io.h>

namespace basalt {

/// Offline reader for the RoboCap rig: the four coverage cameras
/// (dataset type "robocap") or the front stereo pair ("robocap-stereo").
///
/// Frames are downscaled by `downscale` during decode (default 2: the native
/// 1920x1080 streams become 960x540). VIO throughput is bounded by feature
/// detection, which scales with pixel area; the shipped calibrations carry
/// matching downscaled intrinsics. Pass 1 to keep the native resolution.
class RobocapIO final : public DatasetIoInterface {
 public:
  enum class CameraSet { kCoverage, kStereo };

  explicit RobocapIO(CameraSet camera_set = CameraSet::kCoverage, int downscale = 2);
  void read(const std::string& path) override;
  void reset() override;
  VioDatasetPtr get_data() override;

 private:
  CameraSet camera_set_;
  int downscale_;
  VioDatasetPtr data_;
};

}  // namespace basalt
