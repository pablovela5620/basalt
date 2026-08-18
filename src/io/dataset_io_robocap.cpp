/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2026, Pablo Vela.
All rights reserved.
*/

#include <basalt/io/dataset_io_robocap.h>

#include <sqlite3.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/mathematics.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace basalt {
namespace {

namespace fs = std::filesystem;

constexpr int64_t kFramesetToleranceNs = 1'000'000;
constexpr int64_t kCameraToImuOffsetNs = 14'902'432;
constexpr double kAccelScale = 0.001197101;
constexpr double kGyroScale = 0.000266316;

struct CameraSpec {
  int device;
  const char* position;
};

const std::vector<CameraSpec>& coverage_cameras() {
  static const std::vector<CameraSpec> cameras = {
      CameraSpec{4, "left"}, CameraSpec{1, "left-front"},
      CameraSpec{5, "right-front"}, CameraSpec{3, "right"}};
  return cameras;
}

const std::vector<CameraSpec>& stereo_cameras() {
  static const std::vector<CameraSpec> cameras = {CameraSpec{1, "left-front"},
                                                  CameraSpec{5, "right-front"}};
  return cameras;
}

[[noreturn]] void throw_ffmpeg_error(int code, const std::string& operation) {
  std::array<char, AV_ERROR_MAX_STRING_SIZE> buffer{};
  av_strerror(code, buffer.data(), buffer.size());
  throw std::runtime_error(operation + ": " + buffer.data());
}

void check_ffmpeg(int code, const std::string& operation) {
  if (code < 0) { throw_ffmpeg_error(code, operation); }
}

struct FormatContextDeleter {
  void operator()(AVFormatContext* context) const {
    if (context != nullptr) { avformat_close_input(&context); }
  }
};

struct CodecContextDeleter {
  void operator()(AVCodecContext* context) const {
    if (context != nullptr) { avcodec_free_context(&context); }
  }
};

struct PacketDeleter {
  void operator()(AVPacket* packet) const { av_packet_free(&packet); }
};

struct FrameDeleter {
  void operator()(AVFrame* frame) const { av_frame_free(&frame); }
};

using FormatContextPtr = std::unique_ptr<AVFormatContext, FormatContextDeleter>;
using CodecContextPtr = std::unique_ptr<AVCodecContext, CodecContextDeleter>;
using PacketPtr = std::unique_ptr<AVPacket, PacketDeleter>;
using FramePtr = std::unique_ptr<AVFrame, FrameDeleter>;

struct VideoFrameReference {
  std::shared_ptr<const fs::path> path;
  int64_t pts = 0;
  int64_t timestamp_ns = 0;
};

using Frameset = std::vector<VideoFrameReference>;

struct RawImuSample {
  int64_t timestamp_ns = 0;
  Eigen::Vector3d value = Eigen::Vector3d::Zero();
};

int camera_index_for_device(int device, const std::vector<CameraSpec>& cameras) {
  const auto iterator =
      std::find_if(cameras.begin(), cameras.end(),
                   [device](const CameraSpec& camera) {
                     return camera.device == device;
                   });
  return iterator == cameras.end()
             ? -1
             : static_cast<int>(std::distance(cameras.begin(), iterator));
}

int64_t absolute_difference(int64_t first, int64_t second) {
  return first >= second ? first - second : second - first;
}

FormatContextPtr open_format_context(const fs::path& path) {
  AVFormatContext* raw_context = nullptr;
  check_ffmpeg(avformat_open_input(&raw_context, path.c_str(), nullptr, nullptr), "Open " + path.string());
  FormatContextPtr context(raw_context);
  check_ffmpeg(avformat_find_stream_info(context.get(), nullptr), "Read stream info from " + path.string());
  return context;
}

int find_video_stream(AVFormatContext* context, const fs::path& path) {
  const int stream_index = av_find_best_stream(context, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  check_ffmpeg(stream_index, "Find video stream in " + path.string());
  return stream_index;
}

int64_t read_comment_timestamp_us(AVFormatContext* context, int stream_index, const fs::path& path) {
  AVDictionaryEntry* comment = av_dict_get(context->metadata, "comment", nullptr, 0);
  if (comment == nullptr) { comment = av_dict_get(context->streams[stream_index]->metadata, "comment", nullptr, 0); }
  if (comment == nullptr) { throw std::runtime_error("Missing absolute timestamp comment in " + path.string()); }
  try {
    return std::stoll(comment->value);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid absolute timestamp comment in " + path.string());
  }
}

std::vector<VideoFrameReference> index_video(const fs::path& path) {
  FormatContextPtr context = open_format_context(path);
  const int stream_index = find_video_stream(context.get(), path);
  const int64_t first_timestamp_us = read_comment_timestamp_us(context.get(), stream_index, path);
  const AVRational time_base = context->streams[stream_index]->time_base;
  PacketPtr packet(av_packet_alloc());
  if (packet == nullptr) { throw std::bad_alloc(); }
  const auto shared_path = std::make_shared<const fs::path>(path);

  std::vector<VideoFrameReference> frames;
  while (av_read_frame(context.get(), packet.get()) >= 0) {
    if (packet->stream_index == stream_index && packet->pts != AV_NOPTS_VALUE) {
      const int64_t relative_ns = av_rescale_q(packet->pts, time_base, AVRational{1, 1'000'000'000});
      frames.push_back(
          {shared_path, packet->pts, first_timestamp_us * 1000 + relative_ns});
    }
    av_packet_unref(packet.get());
  }
  if (frames.empty()) { throw std::runtime_error("No timestamped video packets in " + path.string()); }
  return frames;
}

class VideoDecoder {
 public:
  ~VideoDecoder() {
    if (scale_context_ != nullptr) { sws_freeContext(scale_context_); }
  }

  std::shared_ptr<ManagedImage<uint16_t>> decode(const VideoFrameReference& reference) {
    if (*reference.path != path_ || reference.pts < last_pts_) {
      open(*reference.path);
    }
    if (reference.pts == last_pts_ && last_image_ != nullptr) { return last_image_; }

    for (;;) {
      const int receive_result = avcodec_receive_frame(codec_context_.get(), frame_.get());
      if (receive_result == 0) {
        const int64_t frame_pts = frame_->best_effort_timestamp;
        if (frame_pts >= reference.pts) {
          if (frame_pts != reference.pts) {
            throw std::runtime_error("Decoder skipped requested PTS in " +
                                     reference.path->string());
          }
          last_pts_ = frame_pts;
          last_image_ = convert_frame(frame_.get());
          av_frame_unref(frame_.get());
          return last_image_;
        }
        av_frame_unref(frame_.get());
        continue;
      }
      if (receive_result == AVERROR_EOF) { throw std::runtime_error("Reached video EOF before requested frame"); }
      if (receive_result != AVERROR(EAGAIN)) { throw_ffmpeg_error(receive_result, "Decode video frame"); }

      bool sent_packet = false;
      while (!sent_packet) {
        const int read_result = av_read_frame(format_context_.get(), packet_.get());
        if (read_result == AVERROR_EOF) {
          check_ffmpeg(avcodec_send_packet(codec_context_.get(), nullptr), "Flush video decoder");
          sent_packet = true;
        } else {
          check_ffmpeg(read_result, "Read video packet");
          if (packet_->stream_index == stream_index_) {
            check_ffmpeg(avcodec_send_packet(codec_context_.get(), packet_.get()), "Send video packet to decoder");
            sent_packet = true;
          }
          av_packet_unref(packet_.get());
        }
      }
    }
  }

 private:
  void open(const fs::path& path) {
    format_context_ = open_format_context(path);
    stream_index_ = find_video_stream(format_context_.get(), path);
    AVCodecParameters* parameters = format_context_->streams[stream_index_]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(parameters->codec_id);
    if (codec == nullptr) { throw std::runtime_error("No decoder for " + path.string()); }
    CodecContextPtr codec_context(avcodec_alloc_context3(codec));
    if (codec_context == nullptr) { throw std::bad_alloc(); }
    check_ffmpeg(avcodec_parameters_to_context(codec_context.get(), parameters), "Copy video codec parameters");
    check_ffmpeg(avcodec_open2(codec_context.get(), codec, nullptr), "Open video decoder");
    codec_context_ = std::move(codec_context);
    packet_.reset(av_packet_alloc());
    frame_.reset(av_frame_alloc());
    if (packet_ == nullptr || frame_ == nullptr) { throw std::bad_alloc(); }
    path_ = path;
    last_pts_ = std::numeric_limits<int64_t>::min();
    last_image_.reset();
  }

  std::shared_ptr<ManagedImage<uint16_t>> convert_frame(const AVFrame* frame) {
    scale_context_ = sws_getCachedContext(scale_context_, frame->width, frame->height,
                                          static_cast<AVPixelFormat>(frame->format), frame->width, frame->height,
                                          AV_PIX_FMT_GRAY8, SWS_POINT, nullptr, nullptr, nullptr);
    if (scale_context_ == nullptr) { throw std::runtime_error("Create grayscale conversion context"); }
    gray_.resize(static_cast<size_t>(frame->width) * frame->height);
    std::array<uint8_t*, 4> destination_data = {gray_.data(), nullptr,
                                                nullptr, nullptr};
    std::array<int, 4> destination_linesize = {frame->width, 0, 0, 0};
    const int rows = sws_scale(scale_context_, frame->data, frame->linesize, 0, frame->height,
                               destination_data.data(), destination_linesize.data());
    if (rows != frame->height) { throw std::runtime_error("Convert complete video frame to grayscale"); }
    auto image = std::make_shared<ManagedImage<uint16_t>>(frame->width, frame->height);
    for (size_t index = 0; index < gray_.size(); ++index) {
      image->ptr[index] = static_cast<uint16_t>(gray_[index]) << 8;
    }
    return image;
  }

  fs::path path_;
  int stream_index_ = -1;
  int64_t last_pts_ = std::numeric_limits<int64_t>::min();
  FormatContextPtr format_context_;
  CodecContextPtr codec_context_;
  PacketPtr packet_;
  FramePtr frame_;
  SwsContext* scale_context_ = nullptr;
  std::vector<uint8_t> gray_;
  std::shared_ptr<ManagedImage<uint16_t>> last_image_;
};

void append_imu_table(sqlite3* database, const char* table,
                      std::vector<RawImuSample>& samples) {
  const std::string sql = std::string("SELECT x,y,z,timestamp FROM ") + table + " ORDER BY timestamp";
  sqlite3_stmt* raw_statement = nullptr;
  if (sqlite3_prepare_v2(database, sql.c_str(), -1, &raw_statement, nullptr) != SQLITE_OK) {
    throw std::runtime_error("Prepare RoboCap IMU query: " + std::string(sqlite3_errmsg(database)));
  }
  std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> statement(raw_statement, sqlite3_finalize);
  for (;;) {
    const int result = sqlite3_step(statement.get());
    if (result == SQLITE_DONE) { break; }
    if (result != SQLITE_ROW) { throw std::runtime_error("Read RoboCap IMU row"); }
    RawImuSample sample;
    sample.value = Eigen::Vector3d(sqlite3_column_int(statement.get(), 0), sqlite3_column_int(statement.get(), 1),
                                   sqlite3_column_int(statement.get(), 2));
    sample.timestamp_ns = sqlite3_column_int64(statement.get(), 3);
    samples.push_back(sample);
  }
}

void append_imu_database(const fs::path& path, std::vector<RawImuSample>& gyro, std::vector<RawImuSample>& accel) {
  sqlite3* raw_database = nullptr;
  if (sqlite3_open_v2(path.c_str(), &raw_database, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    const std::string message = raw_database == nullptr ? "unknown SQLite error" : sqlite3_errmsg(raw_database);
    if (raw_database != nullptr) { sqlite3_close(raw_database); }
    throw std::runtime_error("Open " + path.string() + ": " + message);
  }
  std::unique_ptr<sqlite3, decltype(&sqlite3_close)> database(raw_database, sqlite3_close);
  append_imu_table(database.get(), "gyro_data", gyro);
  append_imu_table(database.get(), "acc_data", accel);
}

void sort_and_deduplicate(std::vector<RawImuSample>& samples) {
  std::sort(samples.begin(), samples.end(),
            [](const RawImuSample& first, const RawImuSample& second) { return first.timestamp_ns < second.timestamp_ns; });
  samples.erase(std::unique(samples.begin(), samples.end(), [](const RawImuSample& first, const RawImuSample& second) {
                  return first.timestamp_ns == second.timestamp_ns;
                }),
                samples.end());
}

class RobocapVioDataset final : public VioDataset {
 public:
  explicit RobocapVioDataset(const std::vector<CameraSpec>& cameras) : cameras_(cameras), decoders_(cameras.size()) {
    for (std::unique_ptr<VideoDecoder>& decoder : decoders_) { decoder = std::make_unique<VideoDecoder>(); }
  }

  size_t get_num_cams() const override { return cameras_.size(); }
  std::vector<int64_t>& get_image_timestamps() override { return image_timestamps_; }
  const Eigen::aligned_vector<AccelData>& get_accel_data() const override { return accel_data_; }
  const Eigen::aligned_vector<GyroData>& get_gyro_data() const override { return gyro_data_; }
  const std::vector<int64_t>& get_gt_timestamps() const override { return gt_timestamps_; }
  const Eigen::aligned_vector<Sophus::SE3d>& get_gt_pose_data() const override { return gt_pose_data_; }
  int64_t get_mocap_to_imu_offset_ns() const override { return 0; }

  std::vector<ImageData> get_image_data(int64_t timestamp_ns) override {
    const auto iterator = framesets_.find(timestamp_ns);
    if (iterator == framesets_.end()) { return {}; }
    std::vector<ImageData> images(cameras_.size());
    for (size_t camera = 0; camera < cameras_.size(); ++camera) {
      images[camera].img = decoders_[camera]->decode(iterator->second[camera]);
    }
    return images;
  }

  void load(const fs::path& path) {
    std::vector<std::vector<VideoFrameReference>> camera_frames(cameras_.size());
    std::vector<fs::path> imu_databases;
    const std::regex video_pattern(R"(video_dev([0-9]+)_session[0-9]+_segment[0-9]+_([^.]+)\.mp4)");
    const std::regex imu_pattern(R"(IMUWriter_dev0_session[0-9]+_segment[0-9]+\.db)");

    if (!fs::is_directory(path)) { throw std::runtime_error("RoboCap dataset path is not a directory: " + path.string()); }
    for (const fs::directory_entry& entry : fs::directory_iterator(path)) {
      if (!entry.is_regular_file()) { continue; }
      const std::string filename = entry.path().filename().string();
      std::smatch match;
      if (std::regex_match(filename, match, video_pattern)) {
        const int device = std::stoi(match[1].str());
        const int camera = camera_index_for_device(device, cameras_);
        if (camera < 0) { continue; }
        if (match[2].str() != cameras_[static_cast<size_t>(camera)].position) {
          throw std::runtime_error("Unexpected RoboCap camera position in " + filename);
        }
        std::vector<VideoFrameReference> indexed = index_video(entry.path());
        camera_frames[static_cast<size_t>(camera)].insert(
            camera_frames[static_cast<size_t>(camera)].end(),
            std::make_move_iterator(indexed.begin()),
            std::make_move_iterator(indexed.end()));
      } else if (std::regex_match(filename, imu_pattern)) {
        imu_databases.push_back(entry.path());
      }
    }
    for (size_t camera = 0; camera < cameras_.size(); ++camera) {
      if (camera_frames[camera].empty()) {
        throw std::runtime_error("Missing RoboCap camera " +
                                 std::string(cameras_[camera].position));
      }
      std::sort(camera_frames[camera].begin(), camera_frames[camera].end(),
                [](const VideoFrameReference& first, const VideoFrameReference& second) {
                  return first.timestamp_ns < second.timestamp_ns;
                });
    }
    build_framesets(camera_frames);
    load_imu(imu_databases);
  }

 private:
  void build_framesets(const std::vector<std::vector<VideoFrameReference>>& camera_frames) {
    const size_t num_cameras = cameras_.size();
    int64_t overlap_start = camera_frames[0].front().timestamp_ns;
    int64_t overlap_end = camera_frames[0].back().timestamp_ns;
    for (size_t camera = 1; camera < num_cameras; ++camera) {
      overlap_start =
          std::max(overlap_start, camera_frames[camera].front().timestamp_ns);
      overlap_end =
          std::min(overlap_end, camera_frames[camera].back().timestamp_ns);
    }
    std::vector<size_t> cursors(num_cameras, 0);
    size_t interior_anchors = 0;
    size_t interior_drops = 0;
    int64_t maximum_skew_ns = 0;

    for (const VideoFrameReference& anchor : camera_frames[0]) {
      if (anchor.timestamp_ns >= overlap_start && anchor.timestamp_ns <= overlap_end) { ++interior_anchors; }
      Frameset frameset(num_cameras);
      frameset[0] = anchor;
      std::vector<size_t> selected = cursors;
      bool complete = true;
      for (size_t camera = 1; camera < num_cameras; ++camera) {
        size_t index = cursors[camera];
        const std::vector<VideoFrameReference>& frames = camera_frames[camera];
        if (index >= frames.size()) {
          complete = false;
          break;
        }
        while (index + 1 < frames.size() &&
               absolute_difference(frames[index + 1].timestamp_ns, anchor.timestamp_ns) <=
                   absolute_difference(frames[index].timestamp_ns, anchor.timestamp_ns)) {
          ++index;
        }
        if (absolute_difference(frames[index].timestamp_ns, anchor.timestamp_ns) > kFramesetToleranceNs) {
          if (frames[index].timestamp_ns < anchor.timestamp_ns) { cursors[camera] = index + 1; }
          complete = false;
          break;
        }
        frameset[camera] = frames[index];
        selected[camera] = index;
      }
      if (!complete) {
        if (anchor.timestamp_ns >= overlap_start && anchor.timestamp_ns <= overlap_end) { ++interior_drops; }
        continue;
      }
      for (size_t camera = 1; camera < num_cameras; ++camera) { cursors[camera] = selected[camera] + 1; }

      std::vector<int64_t> source_timestamps(num_cameras);
      for (size_t camera = 0; camera < num_cameras; ++camera) { source_timestamps[camera] = frameset[camera].timestamp_ns; }
      std::sort(source_timestamps.begin(), source_timestamps.end());
      maximum_skew_ns = std::max(maximum_skew_ns, source_timestamps.back() - source_timestamps.front());
      const size_t middle = num_cameras / 2;
      const int64_t median_timestamp_ns =
          num_cameras % 2 == 1
              ? source_timestamps[middle]
              : source_timestamps[middle - 1] + (source_timestamps[middle] - source_timestamps[middle - 1]) / 2;
      const int64_t timestamp_ns = median_timestamp_ns + kCameraToImuOffsetNs;
      if (!image_timestamps_.empty() && timestamp_ns <= image_timestamps_.back()) {
        throw std::runtime_error("RoboCap frameset timestamps are not strictly increasing");
      }
      image_timestamps_.push_back(timestamp_ns);
      framesets_.emplace(timestamp_ns, std::move(frameset));
    }

    const size_t allowed_drops = std::max<size_t>(1, static_cast<size_t>(std::ceil(interior_anchors * 0.001)));
    if (interior_drops > allowed_drops) { throw std::runtime_error("Excessive incomplete interior RoboCap framesets"); }
    if (image_timestamps_.empty()) { throw std::runtime_error("No complete RoboCap framesets"); }
    std::cout << "RoboCap: " << image_timestamps_.size() << " complete framesets, " << interior_drops
              << " interior drops, max skew " << maximum_skew_ns << " ns" << std::endl;
  }

  void load_imu(std::vector<fs::path> imu_databases) {
    if (imu_databases.empty()) { throw std::runtime_error("Missing RoboCap middle IMU database"); }
    std::sort(imu_databases.begin(), imu_databases.end());
    std::vector<RawImuSample> raw_gyro;
    std::vector<RawImuSample> raw_accel;
    for (const fs::path& database : imu_databases) { append_imu_database(database, raw_gyro, raw_accel); }
    sort_and_deduplicate(raw_gyro);
    sort_and_deduplicate(raw_accel);
    if (raw_gyro.empty() || raw_accel.size() < 2) { throw std::runtime_error("RoboCap middle IMU has no usable data"); }

    gyro_data_.reserve(raw_gyro.size());
    accel_data_.reserve(raw_gyro.size());
    size_t accel_index = 0;
    for (const RawImuSample& gyro : raw_gyro) {
      while (accel_index + 1 < raw_accel.size() && raw_accel[accel_index + 1].timestamp_ns < gyro.timestamp_ns) {
        ++accel_index;
      }
      if (gyro.timestamp_ns < raw_accel[accel_index].timestamp_ns || accel_index + 1 >= raw_accel.size()) { continue; }
      const RawImuSample& before = raw_accel[accel_index];
      const RawImuSample& after = raw_accel[accel_index + 1];
      const double interval = static_cast<double>(after.timestamp_ns - before.timestamp_ns);
      const double alpha = interval == 0.0 ? 0.0 : static_cast<double>(gyro.timestamp_ns - before.timestamp_ns) / interval;
      const Eigen::Vector3d interpolated_accel = before.value + alpha * (after.value - before.value);

      gyro_data_.emplace_back();
      gyro_data_.back().timestamp_ns = gyro.timestamp_ns;
      gyro_data_.back().data = gyro.value * kGyroScale;
      accel_data_.emplace_back();
      accel_data_.back().timestamp_ns = gyro.timestamp_ns;
      accel_data_.back().data = interpolated_accel * kAccelScale;
    }
    if (gyro_data_.empty() || gyro_data_.size() != accel_data_.size()) {
      throw std::runtime_error("RoboCap IMU normalization did not produce paired samples");
    }
    std::cout << "RoboCap: " << gyro_data_.size() << " paired middle-IMU samples" << std::endl;
  }

  std::vector<CameraSpec> cameras_;
  std::vector<int64_t> image_timestamps_;
  std::map<int64_t, Frameset> framesets_;
  Eigen::aligned_vector<AccelData> accel_data_;
  Eigen::aligned_vector<GyroData> gyro_data_;
  std::vector<int64_t> gt_timestamps_;
  Eigen::aligned_vector<Sophus::SE3d> gt_pose_data_;
  std::vector<std::unique_ptr<VideoDecoder>> decoders_;
};

}  // namespace

RobocapIO::RobocapIO(CameraSet camera_set) : camera_set_(camera_set) {}

void RobocapIO::read(const std::string& path) {
  auto data = std::make_shared<RobocapVioDataset>(camera_set_ == CameraSet::kStereo ? stereo_cameras()
                                                                                    : coverage_cameras());
  data->load(path);
  data_ = std::move(data);
}

void RobocapIO::reset() { data_.reset(); }

VioDatasetPtr RobocapIO::get_data() { return data_; }

}  // namespace basalt
