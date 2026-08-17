#include <basalt/utils/rerun_offline_adapter.h>

#include "gtest/gtest.h"

TEST(RerunOfflineAdapterTest, DisabledConfigurationIsAnInertBoundary) {
  basalt::RerunOfflineConfig config;
  basalt::RerunOfflineAdapter adapter(config);

  EXPECT_FALSE(adapter.enabled());
  EXPECT_FALSE(adapter.active());
  adapter.flush();
}
