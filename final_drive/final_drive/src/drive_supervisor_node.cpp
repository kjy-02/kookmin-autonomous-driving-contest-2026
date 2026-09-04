#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int8.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace std::chrono_literals;

namespace {
constexpr float PI = 3.14159265358979323846f;
constexpr uint8_t TRAFFIC_NONE = 0U;
constexpr uint8_t TRAFFIC_RED = 1U;
constexpr uint8_t TRAFFIC_YELLOW = 2U;
constexpr uint8_t TRAFFIC_GREEN = 3U;
constexpr uint8_t TRAFFIC_GREEN_LEFT = 4U;

struct DriveCommand {
  float angle{0.0f};
  float speed{0.0f};
  std::chrono::steady_clock::time_point stamp{};
  bool received{false};
};

struct EntryScanPoint {
  int index{0};
  float range{0.0f};
  float x{0.0f};
  float y{0.0f};
};

struct EntryCluster {
  float x{0.0f};
  float y{0.0f};
};

float normalizeAngle(float angle) {
  while (angle > PI) angle -= 2.0f * PI;
  while (angle < -PI) angle += 2.0f * PI;
  return angle;
}

float moveToward(float current, float target, float step) {
  if (target > current) return std::min(target, current + step);
  return std::max(target, current - step);
}

const char *trafficStateName(uint8_t state) {
  switch (state) {
    case TRAFFIC_RED: return "RED";
    case TRAFFIC_YELLOW: return "YELLOW";
    case TRAFFIC_GREEN: return "GREEN";
    case TRAFFIC_GREEN_LEFT: return "GREEN_LEFT";
    default: return "NONE";
  }
}
}  // namespace

class DriveSupervisor : public rclcpp::Node {
public:
  DriveSupervisor() : Node("drive_supervisor") {
    declare_parameter("command_timeout_sec", 0.30);
    declare_parameter("lidar_timeout_sec", 0.30);
    // The raw bilateral gate is visible before the cone tracker's two-frame
    // confirmation is ready.  Keep only that *bilateral* observation alive
    // long enough for both signals to overlap; one-sided observations never
    // refresh this timer.
    declare_parameter("approach_hold_sec", 0.80);
    declare_parameter("require_lidar_for_motion", true);
    declare_parameter("cone_active_min_speed", 0.50);
    declare_parameter("cone_enter_hold_sec", 0.00);
    declare_parameter("cone_exit_hold_sec", 0.00);
    declare_parameter("cone_reentry_cooldown_sec", 1.00);
    declare_parameter("allow_multiple_cone_sections", true);
    declare_parameter("cone_lane_status_timeout_sec", 0.30);
    declare_parameter("cone_approach_speed", 8.00);
    declare_parameter("steering_step_deg", 12.0);
    declare_parameter("overtake_lidar_ignore_angle_deg", 30.0);
    declare_parameter("overtake_lidar_ignore_hold_sec", 1.0);
    declare_parameter("speed_accel_step", 0.60);
    declare_parameter("speed_decel_step", 1.20);
    declare_parameter("camera_reverse_limit", -10.0);
    declare_parameter("lidar_yaw_deg", 180.0);
    declare_parameter("invert_scan_angle", false);
    declare_parameter("scan_half_angle_deg", 90.0);
    declare_parameter("traffic_topic", "/traffic_light_state");
    declare_parameter("traffic_stop_confirm_frames", 2);
    declare_parameter("traffic_confirm_frames", 3);
    declare_parameter("traffic_timeout_sec", 0.75);
    declare_parameter("require_traffic_for_motion", true);
    declare_parameter("require_green_to_start", true);
    declare_parameter("manual_stop_topic", "/final_drive/manual_stop");
    declare_parameter("manual_stop_timeout_sec", 0.50);
    declare_parameter("require_manual_control_for_motion", true);

    command_timeout_sec_ = get_parameter("command_timeout_sec").as_double();
    lidar_timeout_sec_ = get_parameter("lidar_timeout_sec").as_double();
    approach_hold_sec_ = get_parameter("approach_hold_sec").as_double();
    require_lidar_for_motion_ =
        get_parameter("require_lidar_for_motion").as_bool();
    cone_active_min_speed_ =
        static_cast<float>(get_parameter("cone_active_min_speed").as_double());
    cone_enter_hold_sec_ = get_parameter("cone_enter_hold_sec").as_double();
    cone_exit_hold_sec_ = get_parameter("cone_exit_hold_sec").as_double();
    cone_reentry_cooldown_sec_ =
        get_parameter("cone_reentry_cooldown_sec").as_double();
    allow_multiple_cone_sections_ =
        get_parameter("allow_multiple_cone_sections").as_bool();
    cone_lane_status_timeout_sec_ =
        get_parameter("cone_lane_status_timeout_sec").as_double();
    cone_approach_speed_ =
        static_cast<float>(get_parameter("cone_approach_speed").as_double());
    steering_step_deg_ =
        static_cast<float>(get_parameter("steering_step_deg").as_double());
    overtake_lidar_ignore_angle_deg_ = static_cast<float>(
        get_parameter("overtake_lidar_ignore_angle_deg").as_double());
    overtake_lidar_ignore_hold_sec_ =
        get_parameter("overtake_lidar_ignore_hold_sec").as_double();
    speed_accel_step_ =
        static_cast<float>(get_parameter("speed_accel_step").as_double());
    speed_decel_step_ =
        static_cast<float>(get_parameter("speed_decel_step").as_double());
    camera_reverse_limit_ = static_cast<float>(
        get_parameter("camera_reverse_limit").as_double());
    lidar_yaw_ = static_cast<float>(
        get_parameter("lidar_yaw_deg").as_double() * PI / 180.0);
    invert_scan_angle_ = get_parameter("invert_scan_angle").as_bool();
    scan_half_angle_ = static_cast<float>(
        get_parameter("scan_half_angle_deg").as_double() * PI / 180.0);
    traffic_topic_ = get_parameter("traffic_topic").as_string();
    traffic_stop_confirm_frames_ = std::clamp(
        static_cast<int>(
            get_parameter("traffic_stop_confirm_frames").as_int()),
        1, 3);
    traffic_confirm_frames_ = std::clamp(
        static_cast<int>(get_parameter("traffic_confirm_frames").as_int()),
        2, 3);
    traffic_timeout_sec_ =
        std::max(0.10, get_parameter("traffic_timeout_sec").as_double());
    require_traffic_for_motion_ =
        get_parameter("require_traffic_for_motion").as_bool();
    require_green_to_start_ =
        get_parameter("require_green_to_start").as_bool();
    startup_green_released_ = !require_green_to_start_;
    manual_stop_topic_ = get_parameter("manual_stop_topic").as_string();
    manual_stop_timeout_sec_ =
        std::max(0.10, get_parameter("manual_stop_timeout_sec").as_double());
    require_manual_control_for_motion_ =
        get_parameter("require_manual_control_for_motion").as_bool();
    manual_stop_latched_ = require_manual_control_for_motion_;
    manual_control_armed_ = !require_manual_control_for_motion_;

    command_timeout_sec_ = std::max(0.05, command_timeout_sec_);
    lidar_timeout_sec_ = std::max(0.05, lidar_timeout_sec_);
    approach_hold_sec_ = std::max(0.0, approach_hold_sec_);
    cone_active_min_speed_ = std::max(0.0f, cone_active_min_speed_);
    cone_enter_hold_sec_ = std::max(0.0, cone_enter_hold_sec_);
    cone_exit_hold_sec_ = std::max(0.0, cone_exit_hold_sec_);
    cone_reentry_cooldown_sec_ = std::max(0.0, cone_reentry_cooldown_sec_);
    cone_lane_status_timeout_sec_ =
        std::max(0.05, cone_lane_status_timeout_sec_);
    cone_approach_speed_ = std::max(1.0f, cone_approach_speed_);
    steering_step_deg_ = std::max(1.0f, steering_step_deg_);
    overtake_lidar_ignore_angle_deg_ =
        std::clamp(overtake_lidar_ignore_angle_deg_, 1.0f, 50.0f);
    overtake_lidar_ignore_hold_sec_ =
        std::max(0.0, overtake_lidar_ignore_hold_sec_);
    speed_accel_step_ = std::max(0.05f, speed_accel_step_);
    speed_decel_step_ = std::max(0.05f, speed_decel_step_);
    camera_reverse_limit_ = std::clamp(camera_reverse_limit_, -10.0f, 0.0f);
    scan_half_angle_ = std::clamp(scan_half_angle_, 10.0f * PI / 180.0f,
                                  PI * 0.5f);

    camera_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        "/final_drive/camera_cmd", 10,
        [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
          acceptCommand(*msg, camera_command_, camera_reverse_limit_);
        });
    cone_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        "/final_drive/cone_cmd", 10,
        [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
          acceptCommand(*msg, cone_command_, 0.0f);
        });
    cone_lane_status_sub_ = create_subscription<std_msgs::msg::UInt8>(
        "/final_drive/cone_lane_status", 10,
        std::bind(&DriveSupervisor::coneLaneStatusCallback, this,
                  std::placeholders::_1));
    auto scan_qos = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", scan_qos,
        std::bind(&DriveSupervisor::scanCallback, this, std::placeholders::_1));
    traffic_sub_ = create_subscription<std_msgs::msg::UInt8>(
        traffic_topic_, rclcpp::QoS(10).best_effort(),
        std::bind(&DriveSupervisor::trafficCallback, this,
                  std::placeholders::_1));
    manual_stop_sub_ = create_subscription<std_msgs::msg::Bool>(
        manual_stop_topic_, rclcpp::QoS(1).reliable(),
        std::bind(&DriveSupervisor::manualStopCallback, this,
                  std::placeholders::_1));

    motor_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/xycar_motor", 1);
    cone_status_pub_ =
        create_publisher<std_msgs::msg::UInt8>("/final_drive/cone_status", 1);
    mode_pub_ = create_publisher<std_msgs::msg::String>(
        "/final_drive/mode", rclcpp::QoS(1).reliable().transient_local());
    traffic_state_pub_ = create_publisher<std_msgs::msg::UInt8>(
        "/final_drive/traffic_state",
        rclcpp::QoS(1).reliable().transient_local());
    manual_stop_state_pub_ = create_publisher<std_msgs::msg::Bool>(
        "/final_drive/manual_stop_state",
        rclcpp::QoS(1).reliable().transient_local());
    overtake_lidar_ignore_pub_ = create_publisher<std_msgs::msg::Bool>(
        "/final_drive/overtake_lidar_ignore",
        rclcpp::QoS(1).reliable().transient_local());
    timer_ = create_wall_timer(30ms, std::bind(&DriveSupervisor::controlLoop, this));

    publishMode();
    publishTrafficState();
    publishManualStopState();
    publishOvertakeLidarIgnore(false);
    RCLCPP_INFO(get_logger(),
                "Final drive starts in CAMERA mode: approach speed %.1f, cone enter %.2fs, exit %.2fs, cooldown %.2fs, multi-section=%s; traffic=%s, stop/go confirm=%d/%d frames, green-start=%s",
                cone_approach_speed_, cone_enter_hold_sec_, cone_exit_hold_sec_,
                cone_reentry_cooldown_sec_,
                allow_multiple_cone_sections_ ? "true" : "false",
                 traffic_topic_.c_str(),
                 traffic_stop_confirm_frames_,
                 traffic_confirm_frames_,
                 require_green_to_start_ ? "required" : "disabled");
  }

  ~DriveSupervisor() override { publishMotor(0.0f, 0.0f); }

private:
  enum class Mode { CAMERA, CONE };

  void manualStopCallback(const std_msgs::msg::Bool::SharedPtr message) {
    const bool first_message = !have_manual_stop_message_;
    last_manual_stop_time_ = std::chrono::steady_clock::now();
    have_manual_stop_message_ = true;
    // At startup and after a heartbeat failure, a stale RUN heartbeat must not
    // start the vehicle. Require a STOP heartbeat first, followed by an
    // explicit operator RUN command.
    if (!manual_control_armed_ && !message->data) return;
    if (message->data) manual_control_armed_ = true;

    const bool changed = first_message ||
        manual_stop_latched_ != message->data || manual_stop_timeout_active_;
    manual_stop_latched_ = message->data;
    manual_stop_timeout_active_ = false;
    if (changed) {
      publishManualStopState();
      RCLCPP_WARN(get_logger(), "[KEYBOARD] %s",
                  manual_stop_latched_ ? "STOP" : "RUN");
    }
  }

  void acceptCommand(const std_msgs::msg::Float32MultiArray &msg,
                     DriveCommand &command, float minimum_speed) {
    if (msg.data.size() < 2 || !std::isfinite(msg.data[0]) ||
        !std::isfinite(msg.data[1])) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "Ignoring malformed drive command");
      return;
    }
    command.angle = std::clamp(msg.data[0], -50.0f, 50.0f);
    command.speed = std::max(minimum_speed, msg.data[1]);
    command.stamp = std::chrono::steady_clock::now();
    command.received = true;
  }

  void coneLaneStatusCallback(const std_msgs::msg::UInt8::SharedPtr message) {
    if (message->data > 3U) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "Ignoring invalid cone lane status: %u",
                           static_cast<unsigned int>(message->data));
      return;
    }
    cone_lane_status_ = message->data;
    last_cone_lane_status_time_ = std::chrono::steady_clock::now();
    have_cone_lane_status_ = true;
  }

  bool fresh(const DriveCommand &command,
             const std::chrono::steady_clock::time_point &now) const {
    if (!command.received) return false;
    return std::chrono::duration<double>(now - command.stamp).count() <=
           command_timeout_sec_;
  }

  bool lidarFresh(const std::chrono::steady_clock::time_point &now) const {
    if (!have_scan_) return false;
    return std::chrono::duration<double>(now - last_scan_time_).count() <=
           lidar_timeout_sec_;
  }

  bool trafficFresh(const std::chrono::steady_clock::time_point &now) const {
    if (!have_traffic_message_) return false;
    return std::chrono::duration<double>(now - last_traffic_time_).count() <=
           traffic_timeout_sec_;
  }

  void trafficCallback(const std_msgs::msg::UInt8::SharedPtr message) {
    const uint8_t observed = message->data;
    if (observed > TRAFFIC_GREEN_LEFT) {
      traffic_candidate_state_ = TRAFFIC_NONE;
      traffic_candidate_count_ = 0;
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Ignoring invalid traffic-light state: %u",
          static_cast<unsigned int>(observed));
      return;
    }

    last_traffic_time_ = std::chrono::steady_clock::now();
    have_traffic_message_ = true;

    // NONE means that no lamp is currently visible.  Before the first traffic
    // signal this permits normal driving.  After RED/YELLOW it does not release
    // the stop latch; only a confirmed GREEN state may restart the vehicle.
    if (observed == TRAFFIC_NONE) {
      traffic_candidate_state_ = TRAFFIC_NONE;
      traffic_candidate_count_ = 0;
      return;
    }

    const bool stop_state =
        observed == TRAFFIC_RED || observed == TRAFFIC_YELLOW;
    const int required_frames = stop_state
        ? traffic_stop_confirm_frames_ : traffic_confirm_frames_;
    if (observed == traffic_candidate_state_) {
      traffic_candidate_count_ = std::min(
          traffic_candidate_count_ + 1, required_frames);
    } else {
      traffic_candidate_state_ = observed;
      traffic_candidate_count_ = 1;
    }
    if (traffic_candidate_count_ < required_frames ||
        confirmed_traffic_state_ == observed) {
      return;
    }

    confirmed_traffic_state_ = observed;
    traffic_stop_latched_ = stop_state;
    if (observed == TRAFFIC_GREEN && !startup_green_released_) {
      startup_green_released_ = true;
      RCLCPP_INFO(get_logger(),
                  "[TRAFFIC] initial GREEN confirmed; motion is now permitted");
    }
    publishTrafficState();
    const char *action = traffic_stop_latched_
        ? "STOP"
        : (require_green_to_start_ && !startup_green_released_)
              ? "WAIT INITIAL GREEN"
              : "DRIVE STRAIGHT";
    RCLCPP_INFO(
        get_logger(),
        "[TRAFFIC] confirmed %s after %d frames -> %s",
        trafficStateName(observed), required_frames,
        action);
  }

  bool approachActive(const std::chrono::steady_clock::time_point &now) const {
    if (!have_approach_ || !lidarFresh(now)) return false;
    return std::chrono::duration<double>(now - last_approach_time_).count() <=
           approach_hold_sec_;
  }

  bool coneLaneStatusFresh(
      const std::chrono::steady_clock::time_point &now) const {
    if (!have_cone_lane_status_) return false;
    return std::chrono::duration<double>(
               now - last_cone_lane_status_time_).count() <=
           cone_lane_status_timeout_sec_;
  }

  std::vector<std::vector<EntryScanPoint>> groupScan(
      const sensor_msgs::msg::LaserScan &scan) const {
    std::vector<std::vector<EntryScanPoint>> groups;
    std::vector<EntryScanPoint> current;
    int invalid_run = 0;
    const auto noteInvalid = [&]() {
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
      if (!std::isfinite(range) || range < 0.15f || range > 1.20f ||
          std::abs(angle) > scan_half_angle_) {
        noteInvalid();
        continue;
      }
      EntryScanPoint point;
      point.index = static_cast<int>(index);
      point.range = range;
      point.x = range * std::sin(angle);
      point.y = range * std::cos(angle);
      if (std::abs(point.x) > 0.85f || point.y < 0.0f) {
        noteInvalid();
        continue;
      }
      invalid_run = 0;
      if (!current.empty()) {
        const auto &previous = current.back();
        const float threshold = 0.035f + range * 0.010f;
        if (std::hypot(point.x - previous.x, point.y - previous.y) >
            threshold) {
          groups.push_back(std::move(current));
          current.clear();
        }
      }
      current.push_back(point);
    }
    if (!current.empty()) groups.push_back(std::move(current));

    if (groups.size() >= 2 && groups.front().front().index <= 2 &&
        groups.back().back().index >= static_cast<int>(scan.ranges.size()) - 3) {
      const auto &first = groups.front().front();
      const auto &last = groups.back().back();
      if (std::hypot(first.x - last.x, first.y - last.y) < 0.07f) {
        std::vector<EntryScanPoint> merged = std::move(groups.back());
        merged.insert(merged.end(), groups.front().begin(), groups.front().end());
        groups.front() = std::move(merged);
        groups.pop_back();
      }
    }
    return groups;
  }

  bool makeEntryCluster(const std::vector<EntryScanPoint> &group,
                        EntryCluster &cluster) const {
    if (group.size() < 3) return false;
    float center_x = 0.0f;
    float center_y = 0.0f;
    for (const auto &point : group) {
      center_x += point.x;
      center_y += point.y;
    }
    center_x /= static_cast<float>(group.size());
    center_y /= static_cast<float>(group.size());

    float span = 0.0f;
    for (size_t i = 0; i < group.size(); ++i) {
      for (size_t j = i + 1; j < group.size(); ++j) {
        span = std::max(span, std::hypot(group[i].x - group[j].x,
                                         group[i].y - group[j].y));
      }
    }
    if (span < 0.018f || span > 0.10f) return false;

    float c_xx = 0.0f;
    float c_yy = 0.0f;
    float c_xy = 0.0f;
    for (const auto &point : group) {
      const float dx = point.x - center_x;
      const float dy = point.y - center_y;
      c_xx += dx * dx;
      c_yy += dy * dy;
      c_xy += dx * dy;
    }
    const float count = static_cast<float>(group.size());
    c_xx /= count;
    c_yy /= count;
    c_xy /= count;
    const float trace = c_xx + c_yy;
    const float determinant = c_xx * c_yy - c_xy * c_xy;
    const float eigen_gap = std::sqrt(std::max(
        0.0f, trace * trace * 0.25f - determinant));
    const float lambda1 = trace * 0.5f + eigen_gap;
    const float lambda2 = std::max(1.0e-6f, trace * 0.5f - eigen_gap);
    const float linearity = std::sqrt(lambda1 / lambda2);

    const auto &first = group.front();
    const auto &last = group.back();
    const float chord_x = last.x - first.x;
    const float chord_y = last.y - first.y;
    const float chord = std::hypot(chord_x, chord_y);
    float sagitta = 0.0f;
    float signed_peak = 0.0f;
    if (chord > 0.025f) {
      for (size_t index = 1; index + 1 < group.size(); ++index) {
        const float signed_depth =
            (chord_x * (group[index].y - first.y) -
             chord_y * (group[index].x - first.x)) / chord;
        if (std::abs(signed_depth) > sagitta) {
          sagitta = std::abs(signed_depth);
          signed_peak = signed_depth;
        }
      }
    }
    if (chord > 0.03f && sagitta > 0.003f) {
      const float sensor_side =
          (chord_x * -first.y - chord_y * -first.x) / chord;
      if (sensor_side * signed_peak < 0.0f) return false;
    }
    if (chord > 0.10f && sagitta > 0.0015f) {
      const float fitted_radius =
          chord * chord / (8.0f * sagitta) + sagitta * 0.5f;
      if (fitted_radius > 0.09f) return false;
    }
    if (span > 0.07f && linearity > 4.5f) return false;
    if (span > 0.09f && sagitta < 0.0015f) return false;

    cluster.x = center_x;
    cluster.y = center_y;
    return true;
  }

  std::vector<EntryCluster> detectEntryClusters(
      const sensor_msgs::msg::LaserScan &scan) const {
    std::vector<EntryCluster> clusters;
    for (const auto &group : groupScan(scan)) {
      EntryCluster cluster;
      if (makeEntryCluster(group, cluster)) clusters.push_back(cluster);
    }

    std::vector<bool> wall(clusters.size(), false);
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
          if (std::hypot(clusters[current].x - clusters[next].x,
                         clusters[current].y - clusters[next].y) < 0.16f) {
            visited[next] = true;
            stack.push_back(next);
          }
        }
      }
      if (component.size() >= 3) {
        for (const size_t index : component) wall[index] = true;
      }
    }
    bool expanded = true;
    while (expanded) {
      expanded = false;
      for (size_t i = 0; i < clusters.size(); ++i) {
        if (wall[i]) continue;
        for (size_t j = 0; j < clusters.size(); ++j) {
          if (wall[j] && std::hypot(clusters[i].x - clusters[j].x,
                                    clusters[i].y - clusters[j].y) < 0.22f) {
            wall[i] = true;
            expanded = true;
            break;
          }
        }
      }
    }

    std::vector<EntryCluster> filtered;
    for (size_t index = 0; index < clusters.size(); ++index) {
      if (!wall[index]) filtered.push_back(clusters[index]);
    }
    return filtered;
  }

  bool hasCredibleApproach(const std::vector<EntryCluster> &cones) const {
    for (size_t i = 0; i < cones.size(); ++i) {
      for (size_t j = i + 1; j < cones.size(); ++j) {
        if (cones[i].x * cones[j].x >= 0.0f) continue;
        const float width = std::abs(cones[i].x - cones[j].x);
        const float lateral_center = 0.5f * (cones[i].x + cones[j].x);
        const float mean_forward = 0.5f * (cones[i].y + cones[j].y);
        if (width >= 0.70f && width <= 0.98f &&
            std::abs(cones[i].x) >= 0.25f &&
            std::abs(cones[j].x) >= 0.25f &&
            std::abs(cones[i].x) <= 0.75f &&
            std::abs(cones[j].x) <= 0.75f &&
            std::abs(lateral_center) <= 0.18f &&
            std::abs(cones[i].y - cones[j].y) <= 0.25f &&
            mean_forward >= 0.15f && mean_forward <= 1.20f) {
          return true;
        }
      }
    }
    // Entrance recognition is deliberately bilateral. Same-side rows are
    // never allowed to slow the car or select CONE mode because walls and
    // roadside objects can produce that pattern.
    return false;
  }

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan) {
    const auto callback_time = std::chrono::steady_clock::now();
    last_scan_time_ = callback_time;
    have_scan_ = true;
    const auto cones = detectEntryClusters(*scan);
    if (hasCredibleApproach(cones)) {
      last_approach_time_ = callback_time;
      have_approach_ = true;
    }
  }

  void switchMode(Mode next,
                  const std::chrono::steady_clock::time_point &now) {
    if (mode_ == next) return;
    mode_ = next;
    cone_candidate_active_ = false;
    cone_loss_active_ = false;
    if (mode_ == Mode::CONE) {
      // The approach stage has already reduced camera speed.  Do not clamp to
      // the cone node's still-ramping first command, which caused a needless
      // near-stop during handoff.
      output_speed_ = std::min(output_speed_, cone_approach_speed_);
      RCLCPP_INFO(get_logger(), "[MODE] CAMERA -> CONE");
    } else {
      cone_section_completed_ = true;
      cone_reentry_allowed_at_ = now + std::chrono::duration_cast<
          std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(cone_reentry_cooldown_sec_));
      RCLCPP_INFO(get_logger(), "[MODE] CONE -> CAMERA");
    }
    publishMode();
  }

  void updateMode(const std::chrono::steady_clock::time_point &now) {
    const bool cone_fresh = fresh(cone_command_, now);
    const bool cone_active =
        cone_fresh && cone_command_.speed > cone_active_min_speed_;
    const bool lane_status_fresh = coneLaneStatusFresh(now);
    const bool cone_tracking = lane_status_fresh &&
        (cone_lane_status_ == 1U || cone_lane_status_ == 2U);
    const bool cone_tracking_both =
        lane_status_fresh && cone_lane_status_ == 2U;
    const bool cone_recovering =
        lane_status_fresh && cone_lane_status_ == 3U;
    const bool cone_entry_ready = cone_active &&
        (!lane_status_fresh || cone_tracking_both);
    const bool cone_present = lane_status_fresh
        ? (cone_tracking || cone_recovering)
        : (cone_active || approachActive(now));

    if (mode_ == Mode::CAMERA) {
      cone_loss_active_ = false;
      if ((cone_section_completed_ && !allow_multiple_cone_sections_) ||
          now < cone_reentry_allowed_at_ ||
          !cone_entry_ready ||
          !approachActive(now)) {
        cone_candidate_active_ = false;
        return;
      }
      if (!cone_candidate_active_) {
        cone_candidate_active_ = true;
        cone_candidate_since_ = now;
      }
      if (std::chrono::duration<double>(now - cone_candidate_since_).count() >=
          cone_enter_hold_sec_) {
        switchMode(Mode::CONE, now);
      }
      return;
    }

    cone_candidate_active_ = false;
    if (cone_present) {
      cone_loss_active_ = false;
      return;
    }
    if (!cone_loss_active_) {
      cone_loss_active_ = true;
      cone_loss_since_ = now;
    }
    if (std::chrono::duration<double>(now - cone_loss_since_).count() >=
        cone_exit_hold_sec_) {
      switchMode(Mode::CAMERA, now);
    }
  }

  void controlLoop() {
    const auto now = std::chrono::steady_clock::now();
    updateMode(now);

    const bool manual_stop_stale = have_manual_stop_message_ &&
        std::chrono::duration<double>(now - last_manual_stop_time_).count() >
            manual_stop_timeout_sec_;
    if (manual_stop_stale && !manual_stop_timeout_active_) {
      manual_stop_timeout_active_ = true;
      manual_stop_latched_ = true;
      manual_control_armed_ = false;
      publishManualStopState();
      RCLCPP_ERROR(get_logger(),
                   "Keyboard control heartbeat timed out; latching STOP");
    }

    const DriveCommand &selected =
        mode_ == Mode::CONE ? cone_command_ : camera_command_;
    const bool selected_fresh = fresh(selected, now);
    const bool cone_active = fresh(cone_command_, now) &&
        cone_command_.speed > cone_active_min_speed_;
    const bool lane_status_fresh = coneLaneStatusFresh(now);
    const bool cone_lane_both =
        lane_status_fresh && cone_lane_status_ == 2U;
    const bool cone_mode_lane_active = mode_ == Mode::CONE &&
        lane_status_fresh && cone_lane_status_ >= 1U &&
        cone_lane_status_ <= 3U;
    const bool approach_active = approachActive(now);
    publishConeStatus((cone_lane_both || cone_mode_lane_active ||
                       (!lane_status_fresh && cone_active))
                          ? 2U
                          : approach_active ? 1U : 0U);
    const float target_angle = selected_fresh ? selected.angle : output_angle_;
    float target_speed = selected_fresh ? selected.speed : 0.0f;
    // A credible cone gate is visible before the tracker has enough history
    // to steer.  Keep camera steering during that interval, but shed speed so
    // the confirmed CONE handoff cannot occur at camera-course velocity.
    if (mode_ == Mode::CAMERA && approach_active && target_speed > 0.0f) {
      target_speed = std::min(target_speed, cone_approach_speed_);
    }
    // The camera driver can become ready substantially earlier than the LiDAR
    // driver.  Until cone perception is alive, moving in CAMERA mode would
    // make the upcoming handoff blind.
    const bool waiting_for_lidar = mode_ == Mode::CAMERA &&
        require_lidar_for_motion_ && !lidarFresh(now);
    if (waiting_for_lidar) target_speed = 0.0f;
    const bool waiting_for_traffic =
        require_traffic_for_motion_ && !trafficFresh(now);
    if (waiting_for_traffic || traffic_stop_latched_) target_speed = 0.0f;
    const bool waiting_for_start_green =
        require_green_to_start_ && !startup_green_released_;
    if (waiting_for_start_green) target_speed = 0.0f;
    const bool waiting_for_manual_control =
        require_manual_control_for_motion_ &&
        (!have_manual_stop_message_ || !manual_control_armed_);
    if (waiting_for_manual_control || manual_stop_latched_ || manual_stop_stale) {
      target_speed = 0.0f;
    }

    output_angle_ = moveToward(output_angle_, target_angle, steering_step_deg_);

    // During normal CAMERA driving, a large steering command means the car is
    // in a curve.  Suppress only NEW overtake entry for one second after the
    // most recent |output_angle_| >= threshold event.  If the angle exceeds
    // the threshold again during that second, this deadline is refreshed.
    if (mode_ == Mode::CAMERA &&
        std::abs(output_angle_) >= overtake_lidar_ignore_angle_deg_) {
      overtake_lidar_ignore_until_ = now +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<double>(overtake_lidar_ignore_hold_sec_));
    } else if (mode_ != Mode::CAMERA) {
      // The guard belongs only to CAMERA driving. Do not carry an old curve
      // hold through a CONE section and back into CAMERA mode.
      overtake_lidar_ignore_until_ = std::chrono::steady_clock::time_point{};
    }
    const bool overtake_lidar_ignore = mode_ == Mode::CAMERA &&
        now < overtake_lidar_ignore_until_;
    if (overtake_lidar_ignore != overtake_lidar_ignore_active_) {
      overtake_lidar_ignore_active_ = overtake_lidar_ignore;
      RCLCPP_INFO(
          get_logger(),
          "[OVERTAKE] LiDAR entry gate %s (angle=%.1f deg, threshold=%.1f deg, hold=%.2fs)",
          overtake_lidar_ignore ? "BLOCKED" : "RELEASED", output_angle_,
          overtake_lidar_ignore_angle_deg_, overtake_lidar_ignore_hold_sec_);
    }
    publishOvertakeLidarIgnore(overtake_lidar_ignore);

    if (target_speed < -0.01f) {
      // The camera driver's overtake state uses a bounded -10 command for
      // 0.5 s as active braking. Manual/traffic/lidar stops have already
      // overridden target_speed to zero above and therefore remain dominant.
      output_speed_ = std::max(camera_reverse_limit_, target_speed);
    } else if (target_speed <= 0.01f) {
      // A selected driver's safety stop must never be softened by the mux.
      output_speed_ = 0.0f;
    } else if (output_speed_ < -0.01f) {
      // Release active braking without making the normal acceleration ramp
      // spend another half second crossing the negative-speed range.
      output_speed_ = target_speed;
    } else {
      const float step = target_speed > output_speed_
          ? speed_accel_step_ : speed_decel_step_;
      output_speed_ = moveToward(output_speed_, target_speed, step);
    }
    publishMotor(output_angle_, output_speed_);

    if (!selected_fresh) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "%s command is stale; publishing stop",
          mode_ == Mode::CONE ? "Cone" : "Camera");
    }
    if (waiting_for_lidar) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "LiDAR scan is not ready; holding CAMERA mode at stop");
    }
    if (waiting_for_traffic) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Traffic-light state is not ready; publishing stop");
    }
    if (traffic_stop_latched_) {
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Confirmed %s; holding vehicle at stop",
          trafficStateName(confirmed_traffic_state_));
    }
    if (waiting_for_start_green) {
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Waiting for initial confirmed GREEN; holding vehicle at stop");
    }
    if (manual_stop_latched_ || manual_stop_stale) {
      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Keyboard STOP is active; holding vehicle at stop");
    }
    if (waiting_for_manual_control) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Waiting for keyboard STOP handshake before allowing motion");
    }
  }

  void publishMotor(float angle, float speed) {
    std_msgs::msg::Float32MultiArray message;
    message.data = {
        std::clamp(angle, -50.0f, 50.0f),
        std::max(camera_reverse_limit_, speed),
    };
    motor_pub_->publish(message);
  }

  void publishConeStatus(uint8_t state) {
    std_msgs::msg::UInt8 message;
    message.data = std::min<uint8_t>(state, 2U);
    cone_status_pub_->publish(message);
  }

  void publishMode() {
    std_msgs::msg::String message;
    message.data = mode_ == Mode::CONE ? "CONE" : "CAMERA";
    mode_pub_->publish(message);
  }

  void publishTrafficState() {
    std_msgs::msg::UInt8 message;
    message.data = confirmed_traffic_state_;
    traffic_state_pub_->publish(message);
  }

  void publishManualStopState() {
    std_msgs::msg::Bool message;
    message.data = manual_stop_latched_;
    manual_stop_state_pub_->publish(message);
  }

  void publishOvertakeLidarIgnore(bool ignored) {
    std_msgs::msg::Bool message;
    message.data = ignored;
    overtake_lidar_ignore_pub_->publish(message);
  }

  Mode mode_{Mode::CAMERA};
  DriveCommand camera_command_;
  DriveCommand cone_command_;
  bool cone_candidate_active_{false};
  bool cone_loss_active_{false};
  bool cone_section_completed_{false};
  bool have_scan_{false};
  bool have_approach_{false};
  bool have_cone_lane_status_{false};
  bool have_traffic_message_{false};
  bool traffic_stop_latched_{false};
  bool have_manual_stop_message_{false};
  bool manual_stop_latched_{false};
  bool manual_stop_timeout_active_{false};
  bool manual_control_armed_{false};
  uint8_t traffic_candidate_state_{TRAFFIC_NONE};
  uint8_t confirmed_traffic_state_{TRAFFIC_NONE};
  int traffic_candidate_count_{0};
  std::chrono::steady_clock::time_point cone_candidate_since_{};
  std::chrono::steady_clock::time_point cone_loss_since_{};
  std::chrono::steady_clock::time_point cone_reentry_allowed_at_{};
  std::chrono::steady_clock::time_point last_scan_time_{};
  std::chrono::steady_clock::time_point last_approach_time_{};
  std::chrono::steady_clock::time_point last_cone_lane_status_time_{};
  std::chrono::steady_clock::time_point last_traffic_time_{};
  std::chrono::steady_clock::time_point last_manual_stop_time_{};
  std::chrono::steady_clock::time_point overtake_lidar_ignore_until_{};

  double command_timeout_sec_{0.30};
  double lidar_timeout_sec_{0.30};
  double approach_hold_sec_{0.80};
  bool require_lidar_for_motion_{true};
  float cone_active_min_speed_{0.50f};
  double cone_enter_hold_sec_{0.00};
  double cone_exit_hold_sec_{0.00};
  double cone_reentry_cooldown_sec_{1.00};
  bool allow_multiple_cone_sections_{true};
  double cone_lane_status_timeout_sec_{0.30};
  uint8_t cone_lane_status_{0U};
  float cone_approach_speed_{8.0f};
  float steering_step_deg_{12.0f};
  float overtake_lidar_ignore_angle_deg_{30.0f};
  double overtake_lidar_ignore_hold_sec_{1.0};
  bool overtake_lidar_ignore_active_{false};
  float speed_accel_step_{0.60f};
  float speed_decel_step_{1.20f};
  float camera_reverse_limit_{-10.0f};
  float lidar_yaw_{PI};
  bool invert_scan_angle_{false};
  float scan_half_angle_{PI * 0.5f};
  float output_angle_{0.0f};
  float output_speed_{0.0f};
  std::string traffic_topic_{"/traffic_light_state"};
  int traffic_stop_confirm_frames_{2};
  int traffic_confirm_frames_{3};
  double traffic_timeout_sec_{0.75};
  bool require_traffic_for_motion_{true};
  bool require_green_to_start_{true};
  bool startup_green_released_{false};
  std::string manual_stop_topic_{"/final_drive/manual_stop"};
  double manual_stop_timeout_sec_{0.50};
  bool require_manual_control_for_motion_{true};

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr camera_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr cone_sub_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr cone_lane_status_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr traffic_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr manual_stop_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr motor_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr cone_status_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr mode_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr traffic_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr manual_stop_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr overtake_lidar_ignore_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DriveSupervisor>());
  rclcpp::shutdown();
  return 0;
}
