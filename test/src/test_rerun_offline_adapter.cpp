#include <basalt/utils/rerun_obs_style.h>
#include <basalt/utils/rerun_offline_adapter.h>

#include "gtest/gtest.h"

TEST(RerunOfflineAdapterTest, DisabledConfigurationIsAnInertBoundary) {
  basalt::RerunOfflineConfig config;
  basalt::RerunOfflineAdapter adapter(config);

  EXPECT_FALSE(adapter.enabled());
  EXPECT_FALSE(adapter.active());
  adapter.flush();
}

// Expected values are hand-derived from the Pangolin overlay (do_show_obs in
// vis_utils.cpp): unit_radius = width/96, radius = unit_radius * inv_depth
// clamped to [unit_radius/3, 3*unit_radius], color lerped blue (69,201,255)
// -> pink (255,26,107) over that radius range, alpha 0.15 when the depth
// leaves [1/3 m, 20 m]. Width 640 gives unit_radius 6.667, so the clamp
// range is [2.222, 20].
TEST(DepthCodedRingTest, MidRangeDepthLerpsColorAndScalesRadius) {
  // depth 1 m: radius 6.667, t = 0.25 -> (116, 157, 218), opaque.
  const basalt::ObsRingStyle style = basalt::depth_coded_ring(1.0, 640.0);
  EXPECT_NEAR(style.radius, 6.6667F, 1e-3F);
  EXPECT_EQ(style.rgba, 0x749ddaffu);
}

TEST(DepthCodedRingTest, NearLimitIsLargestPinkRing) {
  // depth exactly 1/3 m: radius hits the 20 px cap, still in valid range.
  const basalt::ObsRingStyle style = basalt::depth_coded_ring(3.0, 640.0);
  EXPECT_NEAR(style.radius, 20.0F, 1e-4F);
  EXPECT_EQ(style.rgba, 0xff1a6bffu);
}

TEST(DepthCodedRingTest, TooCloseDepthIsFaintClampedPinkRing) {
  // depth 1/6 m < 1/3 m: clamped -> alpha 0.15 (38/255).
  const basalt::ObsRingStyle style = basalt::depth_coded_ring(6.0, 640.0);
  EXPECT_NEAR(style.radius, 20.0F, 1e-4F);
  EXPECT_EQ(style.rgba, 0xff1a6b26u);
}

TEST(DepthCodedRingTest, TooFarDepthIsFaintSmallestBlueRing) {
  // depth 100 m > 20 m: clamped -> smallest radius, blue, alpha 0.15.
  const basalt::ObsRingStyle style = basalt::depth_coded_ring(0.01, 640.0);
  EXPECT_NEAR(style.radius, 2.2222F, 1e-3F);
  EXPECT_EQ(style.rgba, 0x45c9ff26u);
}

TEST(DepthCodedRingTest, ZeroInverseDepthIsTreatedAsInfinitelyFar) {
  const basalt::ObsRingStyle style = basalt::depth_coded_ring(0.0, 640.0);
  EXPECT_NEAR(style.radius, 2.2222F, 1e-3F);
  EXPECT_EQ(style.rgba, 0x45c9ff26u);
}
