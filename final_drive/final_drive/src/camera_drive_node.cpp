#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace {
constexpr int WIDTH = 640;
constexpr int HEIGHT = 480;
constexpr int ROI_START_ROW = 300;
constexpr int ROI_END_ROW = 380;
constexpr int ROI_HEIGHT = ROI_END_ROW - ROI_START_ROW;
constexpr int L_ROW = 35;
constexpr int VIEW_CENTER = WIDTH / 2;
constexpr float PI = 3.14159265358979323846f;

const cv::Scalar BLUE(255, 0, 0);
const cv::Scalar GREEN(0, 255, 0);
const cv::Scalar RED(0, 0, 255);
const cv::Scalar YELLOW(0, 255, 255);
const cv::Scalar CYAN(255, 255, 0);

float normalizeAngle(float angle) {
  while (angle > PI) angle -= 2.0f * PI;
  while (angle < -PI) angle += 2.0f * PI;
  return angle;
}
}  // namespace

class CameraDriveNode : public rclcpp::Node {
public:
  CameraDriveNode()
      : Node("camera_drive"),
        obstacle_first_detected_time_(now()),
        last_obstacle_seen_time_(now()),
        last_lane_detected_time_(now()) {
    declare_parameter("debug_view", false);
    declare_parameter("lane_loss_stop_sec", 1.0);
    declare_parameter("straight_speed", 23.0);
    declare_parameter("minimum_curve_speed", 10.0);

    declare_parameter("overtake_enabled", true);
    declare_parameter("lidar_yaw_deg", 180.0);
    declare_parameter("invert_scan_angle", false);
    declare_parameter("overtake_forward_min_m", 0.01);
    declare_parameter("overtake_forward_max_m", 1.50);
    declare_parameter("overtake_lateral_limit_m", 0.40);
    declare_parameter("overtake_wall_analysis_forward_max_m", 1.50);
    declare_parameter("overtake_wall_analysis_lateral_limit_m", 0.90);
    declare_parameter("overtake_min_points", 5);
    declare_parameter("overtake_confirm_scans", 2);
    declare_parameter("overtake_vehicle_max_span_m", 0.48);
    declare_parameter("overtake_wall_seed_distance_m", 0.16);
    declare_parameter("overtake_wall_expand_distance_m", 0.22);
    declare_parameter("overtake_wall_reject_span_m", 0.50);
    declare_parameter("overtake_brake_duration_sec", 0.50);
    declare_parameter("overtake_brake_speed", -10.0);
    declare_parameter("overtake_drive_speed", 15.0);
    declare_parameter("overtake_lane_offset_px", 100);
    declare_parameter("overtake_exit_delay_sec", 1.0);
    declare_parameter("overtake_min_entry_speed", 17.0);
    declare_parameter("overtake_straight_confirm_frames", 2);
    declare_parameter("overtake_straight_max_center_bias_px", 25);
    declare_parameter("overtake_straight_max_angle_deg", 8.0);
    declare_parameter("overtake_straight_min_lane_width_px", 300);
    declare_parameter("overtake_straight_max_lane_width_px", 460);

    debug_view_ = get_parameter("debug_view").as_bool();
    lane_loss_stop_sec_ = get_parameter("lane_loss_stop_sec").as_double();
    straight_speed_ =
        static_cast<float>(get_parameter("straight_speed").as_double());
    minimum_curve_speed_ = static_cast<float>(
        get_parameter("minimum_curve_speed").as_double());

    overtake_enabled_ = get_parameter("overtake_enabled").as_bool();
    lidar_yaw_ = static_cast<float>(
        get_parameter("lidar_yaw_deg").as_double() * PI / 180.0);
    invert_scan_angle_ = get_parameter("invert_scan_angle").as_bool();
    overtake_forward_min_ = static_cast<float>(
        get_parameter("overtake_forward_min_m").as_double());
    overtake_forward_max_ = static_cast<float>(
        get_parameter("overtake_forward_max_m").as_double());
    overtake_lateral_limit_ = static_cast<float>(
        get_parameter("overtake_lateral_limit_m").as_double());
    wall_analysis_forward_max_ = static_cast<float>(
        get_parameter("overtake_wall_analysis_forward_max_m").as_double());
    wall_analysis_lateral_limit_ = static_cast<float>(
        get_parameter("overtake_wall_analysis_lateral_limit_m").as_double());
    overtake_min_points_ = static_cast<int>(
        get_parameter("overtake_min_points").as_int());
    overtake_confirm_scans_ = static_cast<int>(
        get_parameter("overtake_confirm_scans").as_int());
    overtake_vehicle_max_span_ = static_cast<float>(
        get_parameter("overtake_vehicle_max_span_m").as_double());
    wall_seed_distance_ = static_cast<float>(
        get_parameter("overtake_wall_seed_distance_m").as_double());
    wall_expand_distance_ = static_cast<float>(
        get_parameter("overtake_wall_expand_distance_m").as_double());
    wall_reject_span_ = static_cast<float>(
        get_parameter("overtake_wall_reject_span_m").as_double());
    overtake_brake_duration_sec_ =
        get_parameter("overtake_brake_duration_sec").as_double();
    overtake_brake_speed_ = static_cast<float>(
        get_parameter("overtake_brake_speed").as_double());
    overtake_drive_speed_ = static_cast<float>(
        get_parameter("overtake_drive_speed").as_double());
    overtake_lane_offset_px_ = static_cast<int>(
        get_parameter("overtake_lane_offset_px").as_int());
    overtake_exit_delay_sec_ =
        get_parameter("overtake_exit_delay_sec").as_double();
    overtake_min_entry_speed_ = static_cast<float>(
        get_parameter("overtake_min_entry_speed").as_double());
    straight_confirm_frames_ = static_cast<int>(
        get_parameter("overtake_straight_confirm_frames").as_int());
    straight_max_center_bias_px_ = static_cast<int>(
        get_parameter("overtake_straight_max_center_bias_px").as_int());
    straight_max_angle_deg_ = static_cast<float>(
        get_parameter("overtake_straight_max_angle_deg").as_double());
    straight_min_lane_width_px_ = static_cast<int>(
        get_parameter("overtake_straight_min_lane_width_px").as_int());
    straight_max_lane_width_px_ = static_cast<int>(
        get_parameter("overtake_straight_max_lane_width_px").as_int());

    lane_loss_stop_sec_ = std::max(0.1, lane_loss_stop_sec_);
    straight_speed_ = std::max(0.0f, straight_speed_);
    minimum_curve_speed_ =
        std::clamp(minimum_curve_speed_, 0.0f, straight_speed_);
    overtake_forward_min_ = std::max(0.0f, overtake_forward_min_);
    overtake_forward_max_ =
        std::max(overtake_forward_min_ + 0.10f, overtake_forward_max_);
    overtake_lateral_limit_ = std::max(0.10f, overtake_lateral_limit_);
    wall_analysis_forward_max_ =
        std::max(overtake_forward_max_, wall_analysis_forward_max_);
    wall_analysis_lateral_limit_ =
        std::max(overtake_lateral_limit_, wall_analysis_lateral_limit_);
    overtake_min_points_ = std::max(1, overtake_min_points_);
    overtake_confirm_scans_ = std::clamp(overtake_confirm_scans_, 1, 5);
    overtake_vehicle_max_span_ =
        std::max(0.20f, overtake_vehicle_max_span_);
    wall_seed_distance_ = std::max(0.05f, wall_seed_distance_);
    wall_expand_distance_ =
        std::max(wall_seed_distance_, wall_expand_distance_);
    wall_reject_span_ =
        std::max(overtake_vehicle_max_span_, wall_reject_span_);
    overtake_brake_duration_sec_ =
        std::max(0.0, overtake_brake_duration_sec_);
    overtake_brake_speed_ = std::clamp(overtake_brake_speed_, -10.0f, 0.0f);
    overtake_drive_speed_ = std::max(0.0f, overtake_drive_speed_);
    overtake_lane_offset_px_ =
        std::clamp(overtake_lane_offset_px_, 20, WIDTH / 2);
    overtake_exit_delay_sec_ = std::max(0.1, overtake_exit_delay_sec_);
    overtake_min_entry_speed_ = std::max(0.0f, overtake_min_entry_speed_);
    // `straight_speed` is often lowered during track tests.  Allow the same
    // launch command to keep overtaking available; the independent camera
    // straight gate still prevents this lower speed threshold from arming in
    // a curve.
    const float usable_entry_speed =
        std::max(minimum_curve_speed_, straight_speed_ - 5.0f);
    if (overtake_min_entry_speed_ > usable_entry_speed) {
      RCLCPP_WARN(
          get_logger(),
          "overtake_min_entry_speed %.1f is too high for straight_speed %.1f; using %.1f so reduced-speed tests can still overtake",
          overtake_min_entry_speed_, straight_speed_, usable_entry_speed);
      overtake_min_entry_speed_ = usable_entry_speed;
    }
    straight_confirm_frames_ = std::clamp(straight_confirm_frames_, 1, 10);
    straight_max_center_bias_px_ =
        std::clamp(straight_max_center_bias_px_, 5, 100);
    straight_max_angle_deg_ =
        std::clamp(straight_max_angle_deg_, 1.0f, 30.0f);
    straight_min_lane_width_px_ =
        std::clamp(straight_min_lane_width_px_, 100, WIDTH - 1);
    straight_max_lane_width_px_ = std::clamp(
        straight_max_lane_width_px_, straight_min_lane_width_px_, WIDTH);

    sensor_cb_group_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    timer_cb_group_ =
        create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions sensor_options;
    sensor_options.callback_group = sensor_cb_group_;

    command_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
        "/final_drive/camera_cmd", 1);
    lane_state_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
        "/final_drive/camera_lane_state", 1);
    overtake_state_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
        "/final_drive/overtake_state", 1);
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/image_raw", 1,
        std::bind(&CameraDriveNode::imageCallback, this,
                  std::placeholders::_1),
        sensor_options);
    lidar_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", rclcpp::SensorDataQoS(),
        std::bind(&CameraDriveNode::lidarCallback, this,
                  std::placeholders::_1),
        sensor_options);

    const auto mode_qos = rclcpp::QoS(1).reliable().transient_local();
    mode_sub_ = create_subscription<std_msgs::msg::String>(
        "/final_drive/mode", mode_qos,
        [this](const std_msgs::msg::String::SharedPtr message) {
          camera_mode_.store(message->data == "CAMERA");
          have_mode_state_.store(true);
        }, sensor_options);
    cone_status_sub_ = create_subscription<std_msgs::msg::UInt8>(
        "/final_drive/cone_status", rclcpp::QoS(1).reliable(),
        [this](const std_msgs::msg::UInt8::SharedPtr message) {
          cone_status_.store(message->data);
          have_cone_status_.store(true);
        }, sensor_options);
    overtake_lidar_ignore_sub_ = create_subscription<std_msgs::msg::Bool>(
        "/final_drive/overtake_lidar_ignore",
        rclcpp::QoS(1).reliable().transient_local(),
        [this](const std_msgs::msg::Bool::SharedPtr message) {
          overtake_lidar_ignored_.store(message->data);
        }, sensor_options);

    timer_ = create_wall_timer(
        30ms, std::bind(&CameraDriveNode::controlLoop, this), timer_cb_group_);

    RCLCPP_INFO(
        get_logger(),
        "Camera driver ready: speed=%.1f..%.1f, overtake=%s, debug=%s",
        minimum_curve_speed_, straight_speed_, overtake_enabled_ ? "on" : "off",
        debug_view_ ? "true" : "false");
  }

  ~CameraDriveNode() override {
    publishCommand(0.0f, 0.0f);
    if (debug_view_) cv::destroyAllWindows();
  }

private:
  struct CandidateLine {
    cv::Vec4i line;
    float slope{0.0f};
    float x_average{0.0f};
    float white_score{0.0f};
  };

  struct LidarPoint {
    int index{0};
    float range{0.0f};
    float lateral{0.0f};
    float forward{0.0f};
  };

  struct LidarCluster {
    std::vector<LidarPoint> points;
    float center_lateral{0.0f};
    float center_forward{0.0f};
    float min_lateral{0.0f};
    float max_lateral{0.0f};
    float min_forward{0.0f};
    float max_forward{0.0f};
    float span{0.0f};
  };

  struct ObstacleSnapshot {
    int left_points{0};
    int right_points{0};
    int wall_points{0};
    float left_min_distance{0.0f};
    float right_min_distance{0.0f};
    int left_confirm_scans{0};
    int right_confirm_scans{0};
    bool left_confirmed{false};
    bool right_confirmed{false};
  };

  struct StraightGateDiagnostics {
    bool found{false};
    bool both_lanes{false};
    bool not_rejected{false};
    bool lane_width_ok{false};
    bool center_bias_ok{false};
    bool steering_ok{false};
    int lane_width{0};
    int center_bias{0};
  };

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr message) {
    try {
      cv::Mat image = cv_bridge::toCvCopy(message, "bgr8")->image;
      std::lock_guard<std::mutex> lock(image_mutex_);
      image_ = std::move(image);
      has_image_ = true;
      image_sequence_++;
    } catch (const cv_bridge::Exception &error) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", error.what());
    }
  }

  std::vector<std::vector<LidarPoint>> groupObstacleScan(
      const sensor_msgs::msg::LaserScan &scan) const {
    std::vector<std::vector<LidarPoint>> groups;
    std::vector<LidarPoint> current;
    int invalid_run = 0;
    const auto note_invalid = [&]() {
      invalid_run++;
      if (invalid_run > 2 && !current.empty()) {
        groups.push_back(std::move(current));
        current.clear();
      }
    };

    for (size_t index = 0; index < scan.ranges.size(); ++index) {
      const float range = scan.ranges[index];
      const float measured = scan.angle_min +
          static_cast<float>(index) * scan.angle_increment;
      const float raw_angle = invert_scan_angle_ ? -measured : measured;
      const float angle = normalizeAngle(raw_angle + lidar_yaw_);
      if (!std::isfinite(range) || range > 4.0f) {
        note_invalid();
        continue;
      }

      LidarPoint point;
      point.index = static_cast<int>(index);
      point.range = range;
      point.lateral = range * std::sin(angle);
      point.forward = range * std::cos(angle);
      // Keep a wider region while grouping.  A long S-curve wall can cross the
      // narrow vehicle ROI only briefly; cropping first makes that wall look
      // like a compact vehicle.  Wall classification therefore uses the full
      // nearby return, and the vehicle ROI is applied only after rejection.
      if (point.forward < overtake_forward_min_ ||
          point.forward > wall_analysis_forward_max_ ||
          std::abs(point.lateral) > wall_analysis_lateral_limit_) {
        note_invalid();
        continue;
      }

      invalid_run = 0;
      if (!current.empty()) {
        const auto &previous = current.back();
        const float gap = std::hypot(
            point.lateral - previous.lateral,
            point.forward - previous.forward);
        const float threshold = 0.035f + range * 0.010f;
        if (gap > threshold) {
          groups.push_back(std::move(current));
          current.clear();
        }
      }
      current.push_back(point);
    }
    if (!current.empty()) groups.push_back(std::move(current));

    if (groups.size() >= 2) {
      auto &first = groups.front();
      auto &last = groups.back();
      const bool touches_start = first.front().index <= 2;
      const bool touches_end =
          last.back().index >= static_cast<int>(scan.ranges.size()) - 3;
      const float edge_gap = std::hypot(
          first.front().lateral - last.back().lateral,
          first.front().forward - last.back().forward);
      if (touches_start && touches_end && edge_gap < 0.07f) {
        std::vector<LidarPoint> joined;
        joined.reserve(first.size() + last.size());
        joined.insert(joined.end(), last.begin(), last.end());
        joined.insert(joined.end(), first.begin(), first.end());
        groups.front() = std::move(joined);
        groups.pop_back();
      }
    }
    return groups;
  }

  LidarCluster makeLidarCluster(std::vector<LidarPoint> points) const {
    LidarCluster cluster;
    cluster.points = std::move(points);
    cluster.min_lateral = std::numeric_limits<float>::max();
    cluster.max_lateral = std::numeric_limits<float>::lowest();
    cluster.min_forward = std::numeric_limits<float>::max();
    cluster.max_forward = std::numeric_limits<float>::lowest();
    float sum_lateral = 0.0f;
    float sum_forward = 0.0f;
    for (const auto &point : cluster.points) {
      sum_lateral += point.lateral;
      sum_forward += point.forward;
      cluster.min_lateral = std::min(cluster.min_lateral, point.lateral);
      cluster.max_lateral = std::max(cluster.max_lateral, point.lateral);
      cluster.min_forward = std::min(cluster.min_forward, point.forward);
      cluster.max_forward = std::max(cluster.max_forward, point.forward);
    }
    const float count = static_cast<float>(cluster.points.size());
    cluster.center_lateral = sum_lateral / count;
    cluster.center_forward = sum_forward / count;
    cluster.span = std::hypot(
        cluster.max_lateral - cluster.min_lateral,
        cluster.max_forward - cluster.min_forward);
    return cluster;
  }

  float clusterDistance(const LidarCluster &a,
                        const LidarCluster &b) const {
    return std::hypot(a.center_lateral - b.center_lateral,
                      a.center_forward - b.center_forward);
  }

  ObstacleSnapshot analyzeObstacleScan(
      const sensor_msgs::msg::LaserScan &scan) const {
    std::vector<LidarCluster> clusters;
    for (auto &group : groupObstacleScan(scan)) {
      if (!group.empty()) clusters.push_back(makeLidarCluster(std::move(group)));
    }

    std::vector<bool> wall(clusters.size(), false);
    for (size_t index = 0; index < clusters.size(); ++index) {
      if (clusters[index].span > overtake_vehicle_max_span_) wall[index] = true;
    }

    std::vector<bool> visited(clusters.size(), false);
    for (size_t seed = 0; seed < clusters.size(); ++seed) {
      if (visited[seed]) continue;
      std::vector<size_t> stack{seed};
      std::vector<size_t> component;
      visited[seed] = true;
      while (!stack.empty()) {
        const size_t current = stack.back();
        stack.pop_back();
        component.push_back(current);
        for (size_t next = 0; next < clusters.size(); ++next) {
          if (visited[next]) continue;
          if (clusterDistance(clusters[current], clusters[next]) <
              wall_seed_distance_) {
            visited[next] = true;
            stack.push_back(next);
          }
        }
      }

      float min_lateral = std::numeric_limits<float>::max();
      float max_lateral = std::numeric_limits<float>::lowest();
      float min_forward = std::numeric_limits<float>::max();
      float max_forward = std::numeric_limits<float>::lowest();
      for (const size_t index : component) {
        min_lateral = std::min(min_lateral, clusters[index].min_lateral);
        max_lateral = std::max(max_lateral, clusters[index].max_lateral);
        min_forward = std::min(min_forward, clusters[index].min_forward);
        max_forward = std::max(max_forward, clusters[index].max_forward);
      }
      const float component_span =
          std::hypot(max_lateral - min_lateral, max_forward - min_forward);
      if (component_span > wall_reject_span_) {
        for (const size_t index : component) wall[index] = true;
      }
    }

    bool expanded = true;
    while (expanded) {
      expanded = false;
      for (size_t index = 0; index < clusters.size(); ++index) {
        if (wall[index]) continue;
        for (size_t other = 0; other < clusters.size(); ++other) {
          if (!wall[other]) continue;
          if (clusterDistance(clusters[index], clusters[other]) <
              wall_expand_distance_) {
            wall[index] = true;
            expanded = true;
            break;
          }
        }
      }
    }

    ObstacleSnapshot snapshot;
    for (size_t index = 0; index < clusters.size(); ++index) {
      const auto &cluster = clusters[index];
      if (wall[index]) {
        snapshot.wall_points += static_cast<int>(cluster.points.size());
        continue;
      }

      int cluster_left_points = 0;
      int cluster_right_points = 0;
      float cluster_left_min = std::numeric_limits<float>::max();
      float cluster_right_min = std::numeric_limits<float>::max();
      for (const auto &point : cluster.points) {
        if (point.forward > overtake_forward_max_ ||
            std::abs(point.lateral) > overtake_lateral_limit_) {
          continue;
        }
        if (point.lateral >= 0.0f) {
          cluster_left_points++;
          cluster_left_min = std::min(cluster_left_min, point.forward);
        } else {
          cluster_right_points++;
          cluster_right_min = std::min(cluster_right_min, point.forward);
        }
      }

      // A vehicle must be one compact cluster.  Do not add unrelated small
      // fragments on the same side until they accidentally pass the threshold.
      if (cluster_left_points > snapshot.left_points) {
        snapshot.left_points = cluster_left_points;
        snapshot.left_min_distance = cluster_left_min;
      }
      if (cluster_right_points > snapshot.right_points) {
        snapshot.right_points = cluster_right_points;
        snapshot.right_min_distance = cluster_right_min;
      }
    }
    return snapshot;
  }

  void lidarCallback(const sensor_msgs::msg::LaserScan::SharedPtr message) {
    ObstacleSnapshot snapshot = analyzeObstacleScan(*message);
    std::lock_guard<std::mutex> lock(lidar_mutex_);
    left_candidate_scans_ = snapshot.left_points >= overtake_min_points_
        ? std::min(left_candidate_scans_ + 1, overtake_confirm_scans_)
        : 0;
    right_candidate_scans_ = snapshot.right_points >= overtake_min_points_
        ? std::min(right_candidate_scans_ + 1, overtake_confirm_scans_)
        : 0;
    snapshot.left_confirm_scans = left_candidate_scans_;
    snapshot.right_confirm_scans = right_candidate_scans_;
    snapshot.left_confirmed =
        left_candidate_scans_ >= overtake_confirm_scans_;
    snapshot.right_confirmed =
        right_candidate_scans_ >= overtake_confirm_scans_;
    obstacle_snapshot_ = snapshot;
    has_lidar_ = true;
  }

  float whiteScore(const cv::Mat &roi, const cv::Vec4i &line) const {
    cv::LineIterator iterator(
        roi, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), 8);
    const int count = iterator.count;
    if (count == 0) return 0.0f;

    const auto pixel_score = [](const cv::Vec3b &pixel) {
      const int blue = pixel[0];
      const int green = pixel[1];
      const int red = pixel[2];
      const int minimum = std::min({red, green, blue});
      const int maximum = std::max({red, green, blue});
      const int difference = maximum - minimum;
      const int intensity = red + green + blue;
      if (minimum >= 120 && difference <= 35) {
        return static_cast<float>(intensity) +
            static_cast<float>(minimum) * 1.5f -
            static_cast<float>(difference) * 4.0f;
      }
      return std::max(
          0.0f, static_cast<float>(intensity) * 0.5f -
                    static_cast<float>(difference) * 4.0f);
    };

    constexpr int offsets[] = {3, 5, 8};
    float left_total = 0.0f;
    float right_total = 0.0f;
    for (int index = 0; index < count; ++index, ++iterator) {
      const cv::Point point = iterator.pos();
      float left_sum = 0.0f;
      float right_sum = 0.0f;
      for (const int offset : offsets) {
        const int left_x = std::clamp(point.x - offset, 0, roi.cols - 1);
        const int right_x = std::clamp(point.x + offset, 0, roi.cols - 1);
        left_sum += pixel_score(roi.at<cv::Vec3b>(point.y, left_x));
        right_sum += pixel_score(roi.at<cv::Vec3b>(point.y, right_x));
      }
      left_total += left_sum / 3.0f;
      right_total += right_sum / 3.0f;
    }
    return std::max(left_total, right_total) / static_cast<float>(count);
  }

  std::pair<float, float> representativeLine(
      const std::vector<CandidateLine> &candidates,
      float target_x) const {
    if (candidates.empty()) return {0.0f, 0.0f};

    std::vector<CandidateLine> white_candidates;
    for (const auto &candidate : candidates) {
      if (candidate.white_score > 350.0f) {
        white_candidates.push_back(candidate);
      }
    }
    const auto &pool = white_candidates.empty() ? candidates : white_candidates;

    float selected_x = pool.front().x_average;
    float minimum_cost = std::numeric_limits<float>::max();
    for (const auto &candidate : pool) {
      const float cost = std::abs(candidate.x_average - target_x) -
                         candidate.white_score * 0.4f;
      if (cost < minimum_cost) {
        minimum_cost = cost;
        selected_x = candidate.x_average;
      }
    }

    float x_sum = 0.0f;
    float y_sum = 0.0f;
    float slope_sum = 0.0f;
    int count = 0;
    for (const auto &candidate : pool) {
      if (std::abs(candidate.x_average - selected_x) > 30.0f) continue;
      x_sum += static_cast<float>(candidate.line[0] + candidate.line[2]);
      y_sum += static_cast<float>(candidate.line[1] + candidate.line[3]);
      slope_sum += candidate.slope;
      count++;
    }
    if (count == 0) return {0.0f, 0.0f};
    const float x_average = x_sum / static_cast<float>(count * 2);
    const float y_average = y_sum / static_cast<float>(count * 2);
    const float slope = slope_sum / static_cast<float>(count);
    return {slope, y_average - slope * x_average};
  }

  std::tuple<bool, bool, int, int, cv::Mat> detectLane(
      const cv::Mat &source, bool left_obstacle, bool right_obstacle,
      float left_distance, float right_distance) {
    if (source.empty()) return {false, false, 0, 0, cv::Mat()};

    cv::Mat image;
    if (source.cols != WIDTH || source.rows != HEIGHT) {
      cv::resize(source, image, cv::Size(WIDTH, HEIGHT));
    } else {
      image = source;
    }
    cv::Mat display = debug_view_ ? image.clone() : cv::Mat();
    const cv::Rect roi_rectangle(0, ROI_START_ROW, WIDTH, ROI_HEIGHT);
    const cv::Mat roi = image(roi_rectangle);
    cv::Mat line_display = debug_view_ ? roi.clone() : cv::Mat();

    bool detect_left = true;
    bool detect_right = true;
    if (left_obstacle && !right_obstacle) {
      detect_left = false;
    } else if (right_obstacle && !left_obstacle) {
      detect_right = false;
    } else if (left_obstacle && right_obstacle) {
      if (left_distance <= right_distance) {
        detect_left = false;
      } else {
        detect_right = false;
      }
    }

    cv::Mat gray;
    cv::Mat blurred;
    cv::Mat edges;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 60, 80);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, PI / 180.0f, 50, 50, 20);
    if (lines.empty()) return {false, false, 0, 0, display};

    std::vector<CandidateLine> left_candidates;
    std::vector<CandidateLine> right_candidates;
    const bool avoiding = left_obstacle || right_obstacle;
    for (const auto &line : lines) {
      const int x1 = line[0];
      const int y1 = line[1];
      const int x2 = line[2];
      const int y2 = line[3];
      const float slope = x2 == x1
          ? 1000.0f
          : static_cast<float>(y2 - y1) / static_cast<float>(x2 - x1);
      if (std::abs(slope) <= 0.15f) continue;

      CandidateLine candidate;
      candidate.line = line;
      candidate.slope = slope;
      candidate.x_average = static_cast<float>(x1 + x2) * 0.5f;
      candidate.white_score = whiteScore(roi, line);
      if (avoiding) {
        if (detect_left && slope < 0.0f) {
          left_candidates.push_back(candidate);
        } else if (detect_right && slope > 0.0f) {
          right_candidates.push_back(candidate);
        }
      } else if (slope < 0.0f && x1 < VIEW_CENTER) {
        left_candidates.push_back(candidate);
      } else if (slope > 0.0f && x1 > VIEW_CENTER) {
        right_candidates.push_back(candidate);
      }
    }

    if (left_candidates.empty() && right_candidates.empty()) {
      return {false, false, 0, 0, display};
    }

    float left_slope = 0.0f;
    float left_intercept = 0.0f;
    float right_slope = 0.0f;
    float right_intercept = 0.0f;
    if (detect_left) {
      std::tie(left_slope, left_intercept) =
          representativeLine(left_candidates, 305.0f);
    }
    if (detect_right) {
      std::tie(right_slope, right_intercept) =
          representativeLine(right_candidates, 335.0f);
    }
    const bool left_detected = left_slope != 0.0f;
    const bool right_detected = right_slope != 0.0f;
    if (!left_detected && !right_detected) {
      return {false, false, 0, 0, display};
    }

    int left_x = left_detected
        ? static_cast<int>((static_cast<float>(L_ROW) - left_intercept) /
                           left_slope)
        : previous_left_x_;
    int right_x = right_detected
        ? static_cast<int>((static_cast<float>(L_ROW) - right_intercept) /
                           right_slope)
        : previous_right_x_;
    if (!left_detected && right_detected) left_x = right_x - 380;
    if (left_detected && !right_detected) right_x = left_x + 380;
    previous_left_x_ = left_x;
    previous_right_x_ = right_x;

    if (debug_view_) {
      for (const auto &candidate : left_candidates) {
        cv::line(line_display,
                 cv::Point(candidate.line[0], candidate.line[1]),
                 cv::Point(candidate.line[2], candidate.line[3]), RED, 2);
      }
      for (const auto &candidate : right_candidates) {
        cv::line(line_display,
                 cv::Point(candidate.line[0], candidate.line[1]),
                 cv::Point(candidate.line[2], candidate.line[3]), YELLOW, 2);
      }
      const auto draw_representative = [&](float slope, float intercept) {
        if (slope == 0.0f) return;
        const int top_x = static_cast<int>(-intercept / slope);
        const int bottom_x =
            static_cast<int>((ROI_HEIGHT - intercept) / slope);
        cv::line(line_display, cv::Point(top_x, 0),
                 cv::Point(bottom_x, ROI_HEIGHT), BLUE, 2);
      };
      draw_representative(left_slope, left_intercept);
      draw_representative(right_slope, right_intercept);
      cv::line(line_display, cv::Point(0, L_ROW),
               cv::Point(WIDTH, L_ROW), YELLOW, 2);
      cv::circle(line_display,
                 cv::Point((left_x + right_x) / 2, L_ROW), 6, RED, -1);
      line_display.copyTo(display(roi_rectangle));
    }
    return {left_detected || right_detected,
            left_detected && right_detected, left_x, right_x, display};
  }

  std::pair<float, float> controlValues(int target_x, bool detected,
                                        float curve_factor) {
    if (!detected) return {8.0f, previous_angle_};

    const float raw_error = static_cast<float>(target_x - VIEW_CENTER);
    const float absolute_error = std::abs(raw_error);
    float dynamic_kp = kp_;
    if (absolute_error <= 15.0f) {
      const float ratio = absolute_error / 15.0f;
      dynamic_kp = kp_ * (0.25f + 0.75f * ratio * ratio);
    } else if (absolute_error > 60.0f) {
      const float ratio =
          std::min(1.0f, (absolute_error - 60.0f) / 60.0f);
      dynamic_kp = kp_ + ratio * (0.50f - kp_);
    }

    const float derivative_error = raw_error - previous_raw_error_;
    previous_raw_error_ = raw_error;
    float raw_angle = dynamic_kp * raw_error + kd_ * derivative_error;
    raw_angle = std::clamp(raw_angle * 0.90f, -maximum_angle_, maximum_angle_);

    const float dynamic_weight =
        angle_weight_ + curve_factor * (0.60f - angle_weight_);
    const bool unwinding = std::abs(raw_angle) < std::abs(previous_angle_) ||
                           raw_angle * previous_angle_ <= 0.0f;
    const float active_weight = unwinding ? 0.80f : dynamic_weight;
    const float angle =
        (1.0f - active_weight) * previous_angle_ + active_weight * raw_angle;
    previous_angle_ = angle;

    const float curve_deceleration = curve_factor * 12.0f;
    const float angle_deceleration = std::abs(angle) * 0.25f;
    const float speed = std::max(
        minimum_curve_speed_,
        straight_speed_ - curve_deceleration - angle_deceleration);
    return {speed, angle};
  }

  void updateStraightGate(bool found, bool both_lanes,
                          int left_x, int right_x, int lane_center,
                          uint64_t image_sequence) {
    if (straight_gate_image_sequence_ == image_sequence) return;
    straight_gate_image_sequence_ = image_sequence;
    const int lane_width = right_x - left_x;
    straight_gate_diagnostics_.found = found;
    straight_gate_diagnostics_.both_lanes = both_lanes;
    // Field retained for bag schema compatibility. Lane-center jumps are now
    // accepted immediately, so this diagnostic is always ready.
    straight_gate_diagnostics_.not_rejected = true;
    straight_gate_diagnostics_.lane_width = lane_width;
    straight_gate_diagnostics_.center_bias = lane_center - VIEW_CENTER;
    straight_gate_diagnostics_.lane_width_ok =
        lane_width >= straight_min_lane_width_px_ &&
        lane_width <= straight_max_lane_width_px_;
    straight_gate_diagnostics_.center_bias_ok =
        std::abs(lane_center - VIEW_CENTER) <= straight_max_center_bias_px_;
    straight_gate_diagnostics_.steering_ok =
        std::abs(previous_angle_) <= straight_max_angle_deg_;
    // Overtake entry follows the friend's proven gate: require two real camera
    // lanes on consecutive images. Lane width, center bias, and previous
    // steering are retained only as rosbag diagnostics; they no longer block
    // a LiDAR-confirmed vehicle. Lane-center jumps are used immediately.
    const bool straight_candidate = found && both_lanes;
    if (straight_candidate) {
      straight_frames_ = std::min(straight_frames_ + 1,
                                  straight_confirm_frames_);
    } else {
      straight_frames_ = 0;
    }
    straight_ready_ = straight_frames_ >= straight_confirm_frames_;
  }

  bool overtakeEntryAllowed() const {
    return overtake_enabled_ && !overtake_lidar_ignored_.load() &&
           have_mode_state_.load() && have_cone_status_.load() &&
           camera_mode_.load() && cone_status_.load() == 0U &&
           straight_ready_ &&
           previous_nominal_speed_ >= overtake_min_entry_speed_;
  }

  int confirmedObstacleSide(const ObstacleSnapshot &obstacle) const {
    if (obstacle.left_confirmed && !obstacle.right_confirmed) return 1;
    if (obstacle.right_confirmed && !obstacle.left_confirmed) return 2;
    if (obstacle.left_confirmed && obstacle.right_confirmed) {
      return obstacle.left_min_distance <= obstacle.right_min_distance ? 1 : 2;
    }
    return 0;
  }

  void startAvoidance(int side) {
    avoid_side_ = side;
    obstacle_first_detected_time_ = now();
    last_obstacle_seen_time_ = now();
    straight_frames_ = 0;
    straight_ready_ = false;
    RCLCPP_WARN(get_logger(),
                "[OVERTAKE] %s vehicle confirmed -> move %s",
                side == 1 ? "LEFT" : "RIGHT",
                side == 1 ? "RIGHT" : "LEFT");
  }

  void resetAvoidance(const char *reason) {
    if (avoid_side_ != 0) {
      RCLCPP_INFO(get_logger(), "[OVERTAKE] finished/reset: %s", reason);
    }
    avoid_side_ = 0;
    straight_frames_ = 0;
    straight_ready_ = false;
  }

  void updateActiveAvoidance(const ObstacleSnapshot &obstacle) {
    if (avoid_side_ == 0) return;
    if (!overtake_enabled_ || !camera_mode_.load() ||
        cone_status_.load() != 0U) {
      resetAvoidance("mode or cone approach changed");
      return;
    }
    const int current_side_points =
        avoid_side_ == 1 ? obstacle.left_points : obstacle.right_points;
    if (current_side_points >= 1) {
      last_obstacle_seen_time_ = now();
      return;
    }
    if ((now() - last_obstacle_seen_time_).seconds() >=
        overtake_exit_delay_sec_) {
      resetAvoidance("vehicle side clear");
    }
  }

  void controlLoop() {
    cv::Mat current_image;
    uint64_t current_image_sequence = 0;
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      if (!has_image_ || image_.empty()) return;
      current_image = image_.clone();
      current_image_sequence = image_sequence_;
    }

    ObstacleSnapshot obstacle;
    {
      std::lock_guard<std::mutex> lock(lidar_mutex_);
      if (has_lidar_) obstacle = obstacle_snapshot_;
    }

    updateActiveAvoidance(obstacle);
    const bool left_avoidance = avoid_side_ == 1;
    const bool right_avoidance = avoid_side_ == 2;
    auto [found, both_lanes, left_x, right_x, display] = detectLane(
        current_image, left_avoidance, right_avoidance,
        obstacle.left_min_distance, obstacle.right_min_distance);
    int lane_center = (left_x + right_x) / 2;

    if (avoid_side_ == 0) {
      updateStraightGate(found, both_lanes, left_x, right_x, lane_center,
                         current_image_sequence);
      const int confirmed_side = confirmedObstacleSide(obstacle);
      if (confirmed_side != 0 && overtakeEntryAllowed()) {
        startAvoidance(confirmed_side);
      } else if (confirmed_side != 0) {
        RCLCPP_DEBUG_THROTTLE(
            get_logger(), *get_clock(), 500,
            "[OVERTAKE] confirmed but entry blocked: mode=%d cone=%d straight=%d speed=%.1f/%.1f lidar_ignore=%d",
            have_mode_state_.load() && camera_mode_.load(),
            have_cone_status_.load() && cone_status_.load() == 0U,
            straight_ready_, previous_nominal_speed_,
            overtake_min_entry_speed_, overtake_lidar_ignored_.load());
      }
    } else {
      straight_frames_ = 0;
      straight_ready_ = false;
    }

    if (found) {
      last_lane_detected_time_ = now();
      publishLaneState(1.0f, left_x, right_x, lane_center);
    } else {
      publishLaneState(0.0f, left_x, right_x, lane_center);
    }

    if (avoid_side_ != 0) {
      const double lane_lost_seconds =
          (now() - last_lane_detected_time_).seconds();
      const int target_x = std::clamp(
          avoid_side_ == 1 ? right_x - overtake_lane_offset_px_
                           : left_x + overtake_lane_offset_px_,
          0, WIDTH - 1);
      const auto [unused_speed, avoid_angle] =
          controlValues(target_x, found, 0.0f);
      (void)unused_speed;
      const double brake_elapsed =
          (now() - obstacle_first_detected_time_).seconds();
      float speed = brake_elapsed < overtake_brake_duration_sec_
          ? overtake_brake_speed_
          : overtake_drive_speed_;
      if (!found && lane_lost_seconds >= lane_loss_stop_sec_) speed = 0.0f;
      publishCommand(avoid_angle, speed);

      if (debug_view_ && !display.empty()) {
        cv::circle(display,
                   cv::Point(target_x, ROI_START_ROW + L_ROW),
                   8, CYAN, -1);
        cv::putText(
            display,
            cv::format("OVERTAKE %s | L:%d R:%d wall:%d",
                       avoid_side_ == 1 ? "RIGHT" : "LEFT",
                       obstacle.left_points, obstacle.right_points,
                       obstacle.wall_points),
            cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2);
        cv::putText(display,
                    cv::format("Target:%d Angle:%.1f Speed:%.1f",
                               target_x, avoid_angle, speed),
                    cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, GREEN, 2);
      }
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 500,
          "[OVERTAKE] side=%s target=%d angle=%.1f speed=%.1f L=%d R=%d wall=%d",
          avoid_side_ == 1 ? "LEFT" : "RIGHT", target_x, avoid_angle, speed,
          obstacle.left_points, obstacle.right_points, obstacle.wall_points);
    } else if (found) {
      const int curve_bias = lane_center - VIEW_CENTER;
      int target_x = lane_center;
      if (std::abs(curve_bias) > 45) {
        target_x += static_cast<int>(static_cast<float>(curve_bias) * 0.30f);
      }
      target_x = std::clamp(target_x, 0, WIDTH - 1);
      const float curve_factor =
          std::min(1.0f, std::abs(static_cast<float>(curve_bias)) / 100.0f);
      const auto [speed, angle] =
          controlValues(target_x, true, curve_factor);
      previous_nominal_speed_ = speed;
      publishCommand(angle, speed);

      if (debug_view_ && !display.empty()) {
        cv::putText(display, cv::format("Error: %d px", target_x - VIEW_CENTER),
                    cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, YELLOW, 2);
        cv::putText(display,
                    cv::format("Angle %.1f | Speed %.1f | Straight %d/%d",
                               angle, speed, straight_frames_,
                               straight_confirm_frames_),
                    cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, GREEN, 2);
      }
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 500,
          "[CAMERA] target=%d angle=%.1f speed=%.1f straight=%s L=%d R=%d wall=%d",
          target_x, angle, speed, straight_ready_ ? "yes" : "no",
          obstacle.left_points, obstacle.right_points, obstacle.wall_points);
    } else {
      const double lost_seconds =
          (now() - last_lane_detected_time_).seconds();
      if (lost_seconds >= lane_loss_stop_sec_) {
        publishCommand(previous_angle_, 0.0f);
      } else {
        const auto [speed, angle] =
            controlValues(VIEW_CENTER, false, 0.0f);
        publishCommand(angle, speed);
      }
    }

    publishOvertakeState(obstacle);
    if (debug_view_ && !display.empty()) {
      cv::imshow("Final camera drive", display);
      cv::waitKey(1);
    }
  }

  void publishCommand(float angle, float speed) {
    std_msgs::msg::Float32MultiArray message;
    message.data = {
        std::clamp(angle, -50.0f, 50.0f),
        std::max(-10.0f, speed),
    };
    command_pub_->publish(message);
  }

  void publishLaneState(float state, int left_x, int right_x, int center_x) {
    std_msgs::msg::Float32MultiArray message;
    message.data = {
        state,
        static_cast<float>(left_x),
        static_cast<float>(right_x),
        static_cast<float>(center_x),
        static_cast<float>(right_x - left_x),
    };
    lane_state_pub_->publish(message);
  }

  void publishOvertakeState(const ObstacleSnapshot &obstacle) {
    float state = 0.0f;
    if (avoid_side_ != 0) {
      state = (now() - obstacle_first_detected_time_).seconds() <
                      overtake_brake_duration_sec_
                  ? 1.0f
                  : 2.0f;
    }
    std_msgs::msg::Float32MultiArray message;
    uint32_t entry_block_mask = 0U;
    if (!overtake_enabled_) entry_block_mask |= 1U << 0;
    if (!have_mode_state_.load() || !camera_mode_.load()) {
      entry_block_mask |= 1U << 1;
    }
    if (!have_cone_status_.load() || cone_status_.load() != 0U) {
      entry_block_mask |= 1U << 2;
    }
    if (!straight_ready_) entry_block_mask |= 1U << 3;
    if (previous_nominal_speed_ < overtake_min_entry_speed_) {
      entry_block_mask |= 1U << 4;
    }
    if (!obstacle.left_confirmed && !obstacle.right_confirmed) {
      entry_block_mask |= 1U << 5;
    }
    if (overtake_lidar_ignored_.load()) entry_block_mask |= 1U << 6;
    message.data = {
        state,
        static_cast<float>(avoid_side_),
        static_cast<float>(obstacle.left_points),
        static_cast<float>(obstacle.right_points),
        obstacle.left_min_distance,
        obstacle.right_min_distance,
        straight_ready_ ? 1.0f : 0.0f,
        static_cast<float>(obstacle.wall_points),
        static_cast<float>(obstacle.left_confirm_scans),
        static_cast<float>(obstacle.right_confirm_scans),
        obstacle.left_confirmed ? 1.0f : 0.0f,
        obstacle.right_confirmed ? 1.0f : 0.0f,
        have_mode_state_.load() && camera_mode_.load() ? 1.0f : 0.0f,
        have_cone_status_.load() && cone_status_.load() == 0U ? 1.0f : 0.0f,
        previous_nominal_speed_ >= overtake_min_entry_speed_ ? 1.0f : 0.0f,
        straight_gate_diagnostics_.found ? 1.0f : 0.0f,
        straight_gate_diagnostics_.both_lanes ? 1.0f : 0.0f,
        straight_gate_diagnostics_.not_rejected ? 1.0f : 0.0f,
        straight_gate_diagnostics_.lane_width_ok ? 1.0f : 0.0f,
        straight_gate_diagnostics_.center_bias_ok ? 1.0f : 0.0f,
        straight_gate_diagnostics_.steering_ok ? 1.0f : 0.0f,
        static_cast<float>(straight_gate_diagnostics_.lane_width),
        static_cast<float>(straight_gate_diagnostics_.center_bias),
        previous_angle_,
        previous_nominal_speed_,
        static_cast<float>(entry_block_mask),
    };
    overtake_state_pub_->publish(message);
  }

  bool debug_view_{false};
  double lane_loss_stop_sec_{1.0};
  float straight_speed_{23.0f};
  float minimum_curve_speed_{10.0f};

  bool overtake_enabled_{true};
  float lidar_yaw_{PI};
  bool invert_scan_angle_{false};
  float overtake_forward_min_{0.01f};
  float overtake_forward_max_{1.00f};
  float overtake_lateral_limit_{0.40f};
  float wall_analysis_forward_max_{1.30f};
  float wall_analysis_lateral_limit_{0.90f};
  int overtake_min_points_{5};
  int overtake_confirm_scans_{2};
  float overtake_vehicle_max_span_{0.48f};
  float wall_seed_distance_{0.16f};
  float wall_expand_distance_{0.22f};
  float wall_reject_span_{0.50f};
  double overtake_brake_duration_sec_{0.50};
  float overtake_brake_speed_{-10.0f};
  float overtake_drive_speed_{5.0f};
  int overtake_lane_offset_px_{100};
  double overtake_exit_delay_sec_{1.0};
  float overtake_min_entry_speed_{17.0f};
  int straight_confirm_frames_{2};
  int straight_max_center_bias_px_{25};
  float straight_max_angle_deg_{8.0f};
  int straight_min_lane_width_px_{300};
  int straight_max_lane_width_px_{460};

  std::mutex image_mutex_;
  cv::Mat image_;
  bool has_image_{false};
  uint64_t image_sequence_{0};
  uint64_t straight_gate_image_sequence_{0};

  std::mutex lidar_mutex_;
  ObstacleSnapshot obstacle_snapshot_;
  bool has_lidar_{false};
  int left_candidate_scans_{0};
  int right_candidate_scans_{0};
  std::atomic<bool> camera_mode_{true};
  std::atomic<uint8_t> cone_status_{0U};
  std::atomic<bool> have_mode_state_{false};
  std::atomic<bool> have_cone_status_{false};
  std::atomic<bool> overtake_lidar_ignored_{false};

  int previous_left_x_{100};
  int previous_right_x_{540};
  int straight_frames_{0};
  bool straight_ready_{false};
  StraightGateDiagnostics straight_gate_diagnostics_;
  int avoid_side_{0};
  float previous_nominal_speed_{20.0f};

  float kp_{0.20f};
  float kd_{1.00f};
  float maximum_angle_{50.0f};
  float angle_weight_{0.92f};
  float previous_angle_{0.0f};
  float previous_raw_error_{0.0f};
  rclcpp::Time obstacle_first_detected_time_;
  rclcpp::Time last_obstacle_seen_time_;
  rclcpp::Time last_lane_detected_time_;

  rclcpp::CallbackGroup::SharedPtr sensor_cb_group_;
  rclcpp::CallbackGroup::SharedPtr timer_cb_group_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr command_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr lane_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr
      overtake_state_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr mode_sub_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr cone_status_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr overtake_lidar_ignore_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CameraDriveNode>();
  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), 4);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
