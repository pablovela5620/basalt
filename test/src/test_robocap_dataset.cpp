/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2026, Pablo Vela.
All rights reserved.
*/

#include <basalt/io/dataset_io.h>

#include <gtest/gtest.h>
#include <sqlite3.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr int64_t kFirstFrameTimestampNs = 1'014'902'432;
constexpr double kAccelScale = 0.001197101;
constexpr double kGyroScale = 0.000266316;

void sqlite_exec(sqlite3* database, const std::string& sql) {
  char* error = nullptr;
  const int result = sqlite3_exec(database, sql.c_str(), nullptr, nullptr, &error);
  if (result != SQLITE_OK) {
    const std::string message = error == nullptr ? "unknown SQLite error" : error;
    sqlite3_free(error);
    throw std::runtime_error(message);
  }
}

class TemporaryRobocapSession {
 public:
  TemporaryRobocapSession() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = fs::temp_directory_path() / ("basalt-robocap-fixture-" + std::to_string(nonce));
    fs::create_directories(path_);

    create_segment(1, 1'000'000);
    create_segment(2, 1'100'000);
    create_imu_database();
  }

  ~TemporaryRobocapSession() { fs::remove_all(path_); }

  const fs::path& path() const { return path_; }

 private:
  void create_segment(int segment, int64_t timestamp_us) const {
    create_video(4, "left", "black", segment, timestamp_us);
    create_video(1, "left-front", "0x404040", segment, timestamp_us);
    create_video(5, "right-front", "0x808080", segment, timestamp_us);
    create_video(3, "right", "white", segment, timestamp_us);
  }

  void create_video(int device, const std::string& position,
                    const std::string& color, int segment,
                    int64_t timestamp_us) const {
    const char* prefix = std::getenv("CONDA_PREFIX");
    if (prefix == nullptr) { throw std::runtime_error("CONDA_PREFIX is required for the fixture FFmpeg"); }
    const fs::path ffmpeg = fs::path(prefix) / "bin" / "ffmpeg";
    const fs::path output =
        path_ / ("video_dev" + std::to_string(device) +
                 "_session18_segment" + std::to_string(segment) + "_" +
                 position + ".mp4");
    const std::string command = "'" + ffmpeg.string() + "' -hide_banner -loglevel error -y -f lavfi -i "
                                "'color=c=" +
                                color + ":s=16x16:r=30:d=0.1' -frames:v 3 -c:v libx264 -bf 0 -pix_fmt yuv420p "
                                        "-video_track_timescale 90000 -metadata comment=" +
                                std::to_string(timestamp_us) + " '" +
                                output.string() + "'";
    if (std::system(command.c_str()) != 0) { throw std::runtime_error("Failed to create H.264 fixture video"); }
  }

  void create_imu_database() const {
    const fs::path database_path = path_ / "IMUWriter_dev0_session18_segment1.db";
    sqlite3* raw_database = nullptr;
    if (sqlite3_open(database_path.c_str(), &raw_database) != SQLITE_OK) {
      throw std::runtime_error("Failed to create fixture IMU database");
    }
    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> database(
        raw_database, sqlite3_close);
    sqlite_exec(database.get(),
                "CREATE TABLE gyro_data (id INTEGER PRIMARY KEY AUTOINCREMENT, imuid_ INTEGER, x INTEGER, y "
                "INTEGER, z INTEGER, timestamp INTEGER);");
    sqlite_exec(database.get(),
                "CREATE TABLE acc_data (id INTEGER PRIMARY KEY AUTOINCREMENT, imuid_ INTEGER, x INTEGER, y "
                "INTEGER, z INTEGER, timestamp INTEGER);");
    sqlite_exec(database.get(),
                "INSERT INTO gyro_data(imuid_,x,y,z,timestamp) VALUES"
                "(0,100,200,300,1000000000),(0,110,210,310,1005000000),(0,120,220,320,1010000000);"
                "INSERT INTO acc_data(imuid_,x,y,z,timestamp) VALUES"
                "(0,1000,2000,3000,999000000),(0,1100,2100,3100,1004000000),"
                "(0,1200,2200,3200,1009000000),(0,1300,2300,3300,1014000000);");
  }

  fs::path path_;
};

TEST(RobocapDatasetIo, ReadsCoverageCamerasAndPairedImuThroughPublicInterface) {
  TemporaryRobocapSession session;
  basalt::DatasetIoInterfacePtr dataset_io = basalt::DatasetIoFactory::getDatasetIo("robocap");
  dataset_io->read(session.path().string());
  basalt::VioDatasetPtr dataset = dataset_io->get_data();

  ASSERT_NE(dataset, nullptr);
  EXPECT_EQ(dataset->get_num_cams(), 4);
  const std::vector<int64_t>& image_timestamps = dataset->get_image_timestamps();
  ASSERT_EQ(image_timestamps.size(), 6);
  EXPECT_EQ(image_timestamps.front(), kFirstFrameTimestampNs);
  EXPECT_LT(image_timestamps[0], image_timestamps[1]);
  EXPECT_LT(image_timestamps[1], image_timestamps[2]);

  const std::vector<basalt::ImageData> images = dataset->get_image_data(image_timestamps.front());
  ASSERT_EQ(images.size(), 4);
  std::vector<uint16_t> first_pixels;
  for (const basalt::ImageData& image : images) {
    ASSERT_NE(image.img, nullptr);
    EXPECT_EQ(image.img->w, 16);
    EXPECT_EQ(image.img->h, 16);
    first_pixels.push_back(image.img->ptr[0]);
  }
  EXPECT_LT(first_pixels[0], first_pixels[1]);
  EXPECT_LT(first_pixels[1], first_pixels[2]);
  EXPECT_LT(first_pixels[2], first_pixels[3]);

  const std::vector<basalt::ImageData> last_images =
      dataset->get_image_data(image_timestamps.back());
  ASSERT_EQ(last_images.size(), 4);
  for (const basalt::ImageData& image : last_images) {
    ASSERT_NE(image.img, nullptr);
    EXPECT_EQ(image.img->w, 16);
    EXPECT_EQ(image.img->h, 16);
  }

  const Eigen::aligned_vector<basalt::GyroData>& gyro = dataset->get_gyro_data();
  const Eigen::aligned_vector<basalt::AccelData>& accel = dataset->get_accel_data();
  ASSERT_EQ(gyro.size(), 3);
  ASSERT_EQ(accel.size(), gyro.size());
  for (size_t index = 0; index < gyro.size(); ++index) { EXPECT_EQ(accel[index].timestamp_ns, gyro[index].timestamp_ns); }
  EXPECT_NEAR(gyro[0].data.x(), 100.0 * kGyroScale, 1e-12);
  EXPECT_NEAR(gyro[0].data.y(), 200.0 * kGyroScale, 1e-12);
  EXPECT_NEAR(gyro[0].data.z(), 300.0 * kGyroScale, 1e-12);
  EXPECT_NEAR(accel[0].data.x(), 1020.0 * kAccelScale, 1e-12);
  EXPECT_NEAR(accel[0].data.y(), 2020.0 * kAccelScale, 1e-12);
  EXPECT_NEAR(accel[0].data.z(), 3020.0 * kAccelScale, 1e-12);
}

}  // namespace
