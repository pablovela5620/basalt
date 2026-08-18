/**
Depth-coded styling for 2D landmark-observation rings, mirroring the Pangolin
GUI overlay (do_show_obs in vis_utils.cpp): ring radius and color both encode
landmark depth. Kept free of the Rerun SDK so it is unit-testable on its own.
*/
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace basalt {

/// One observation ring: radius in image pixels + packed 0xRRGGBBAA color.
struct ObsRingStyle {
  float radius = 0.0F;
  uint32_t rgba = 0;
};

/// Map a landmark's inverse depth to the Pangolin do_show_obs ring style:
/// radius = (cam0_width/96) / depth clamped to [unit/3, 3*unit]; color lerped
/// from blue (far/small) to pink (near/large) over that radius range; alpha
/// drops to 0.15 when the depth leaves the valid [1/3 m, 20 m] range (the
/// GUI's "clamped" marker). `inv_depth <= 0` counts as infinitely far.
inline ObsRingStyle depth_coded_ring(double inv_depth, double cam0_width) {
  constexpr double kMinDepth = 1.0 / 3.0;  // from valid_kp in sqrt_keypoint_vio.cpp
  constexpr double kMaxDepth = 20.0;
  constexpr double kBlue[3] = {69.0, 201.0, 255.0};  // MIN_DEPTH_COLOR_UB
  constexpr double kPink[3] = {255.0, 26.0, 107.0};  // MAX_DEPTH_COLOR_UB

  const double unit_radius = cam0_width / 96.0;
  const double min_radius = unit_radius * kMinDepth;
  const double max_radius = unit_radius / kMinDepth;

  const double depth = inv_depth > 0.0 ? 1.0 / inv_depth : std::numeric_limits<double>::infinity();
  const bool clamped = depth < kMinDepth || depth > kMaxDepth;
  const double radius = std::clamp(unit_radius * std::max(inv_depth, 0.0), min_radius, max_radius);
  const double t = (radius - min_radius) / (max_radius - min_radius);

  const auto channel = [&](int i) {
    return static_cast<uint32_t>(std::lround(kBlue[i] + t * (kPink[i] - kBlue[i])));
  };
  const uint32_t alpha = clamped ? static_cast<uint32_t>(std::lround(0.15 * 255.0)) : 0xffu;

  return ObsRingStyle{static_cast<float>(radius),
                      (channel(0) << 24) | (channel(1) << 16) | (channel(2) << 8) | alpha};
}

}  // namespace basalt
