#include <cstdlib>

#include <basalt/utils/rerun_vit_adapter.h>

#include "gtest/gtest.h"

TEST(RerunVitConfigTest, ReadsSpawnConfigurationFromEnvironment) {
  setenv("BASALT_VIT_RERUN", "spawn", 1);
  setenv("BASALT_VIT_RERUN_IMG_STRIDE", "3", 1);
  setenv("BASALT_VIT_RERUN_IMU", "true", 1);

  const basalt::RerunVitConfig config = basalt::RerunVitConfig::from_environment();

  EXPECT_TRUE(config.enabled);
  EXPECT_TRUE(config.spawn);
  EXPECT_TRUE(config.log_imu);
  EXPECT_EQ(config.image_stride, 3);
  EXPECT_TRUE(config.rrd_path.empty());
  EXPECT_TRUE(config.connect_url.empty());

  unsetenv("BASALT_VIT_RERUN");
  unsetenv("BASALT_VIT_RERUN_IMG_STRIDE");
  unsetenv("BASALT_VIT_RERUN_IMU");
}

TEST(RerunVitAdapterTest, DisabledConfigurationIsAnInertBoundary) {
  basalt::RerunVitConfig config;
  basalt::RerunVitAdapter adapter(config);

  EXPECT_FALSE(adapter.enabled());
  EXPECT_FALSE(adapter.active());
  EXPECT_FALSE(adapter.requires_pose_features());
  adapter.note_frame_timestamp(123);
  adapter.log_imu_sample(123, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  adapter.flush();
}
