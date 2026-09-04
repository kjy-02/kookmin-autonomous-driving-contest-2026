#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace {
constexpr float PI = 3.14159265358979323846f;
constexpr float CONE_SPACING_M = 0.29f;
constexpr float TRACK_WIDTH_M = 0.84f;
constexpr float HALF_WIDTH_M = TRACK_WIDTH_M * 0.5f;

float normalizeAngle(float angle) {
  while (angle > PI) angle -= 2.0f * PI;
  while (angle < -PI) angle += 2.0f * PI;
  return angle;
}

struct Point {
  float x{0.0f};  // +x: vehicle left
  float y{0.0f};  // +y: vehicle forward

  float distanceTo(const Point &other) const {
    return std::hypot(x - other.x, y - other.y);
  }
};

struct ScanPoint : Point {
  int index{0};
  float angle{0.0f};
  float range{0.0f};
};

struct ConeCluster : Point {
  int point_count{0};
  float span{0.0f};
  float linearity{0.0f};
  float sagitta{0.0f};
  bool faces_sensor{true};
};

struct TrackedCone : Point {
  int hits{1};
  int missing{0};
};

struct WallFragmentTrack : Point {
  // Scans elapsed since this track was last supported by a dense wall group.
  // A mere proximity match must not refresh wall memory indefinitely.
  int unsupported_scans{0};
};
}  // namespace

class ConeDriveV2 : public rclcpp::Node {
public:
  ConeDriveV2() : Node("cone_drive_v2") {
    declare_parameter("debug_mode", true);
    declare_parameter("lidar_yaw_deg", 180.0);  // rear-facing sensor
    declare_parameter("invert_scan_angle", false);  // upside-down/mirrored mounting only
    declare_parameter("scan_half_angle_deg", 90.0);  // vehicle-front -90..+90
    declare_parameter("cone_min_dist", 0.15);
    declare_parameter("cone_max_dist", 1.20);
    declare_parameter("max_x_limit", 0.85);
    declare_parameter("min_cluster_points", 3);
    declare_parameter("min_cluster_span", 0.018);
    declare_parameter("cone_base_radius", 0.07);
    declare_parameter("cone_diameter_tolerance", 0.02);
    declare_parameter("cone_radius_fit_tolerance", 0.02);
    declare_parameter("max_cluster_span", 0.16);  // 14 cm cone base + 2 cm scan tolerance
    declare_parameter("max_cone_shape_span", 0.10);
    declare_parameter("max_linearity", 4.5);
    declare_parameter("min_sagitta", 0.0015);
    declare_parameter("wall_fragment_seed_distance", 0.16);
    declare_parameter("wall_fragment_expand_distance", 0.22);
    declare_parameter("wall_fragment_match_distance", 0.18);
    declare_parameter("wall_fragment_min_group", 3);
    declare_parameter("wall_fragment_memory_scans", 4);
    declare_parameter("drive_speed", 9.5);
    declare_parameter("one_side_speed", 6.0);
    declare_parameter("stable_one_side_speed", 7.0);
    declare_parameter("stable_one_side_scans", 3);
    declare_parameter("curve_speed", 7.5);
    declare_parameter("reversal_speed", 5.5);
    declare_parameter("clearance_speed", 5.5);
    declare_parameter("full_speed_angle_deg", 8.0);
    declare_parameter("slow_speed_angle_deg", 28.0);
    declare_parameter("recovery_speed", 3.0);
    declare_parameter("singleton_hold_scans", 5);
    declare_parameter("empty_hold_scans", 2);
    declare_parameter("speed_accel_step", 0.35);
    declare_parameter("speed_decel_step", 0.45);
    declare_parameter("min_lane_width", 0.70);
    declare_parameter("max_lane_width", 0.98);  // 0.90 m course + measurement tolerance
    declare_parameter("vehicle_length", 0.60);
    declare_parameter("vehicle_width", 0.30);
    declare_parameter("wheelbase", 0.30);
    declare_parameter("rear_overhang", 0.15);
    declare_parameter("front_overhang", 0.15);
    declare_parameter("lidar_ahead_of_front_axle", 0.10);
    declare_parameter("body_clearance_margin", 0.08);
    declare_parameter("near_cone_check_forward", 0.30);
    declare_parameter("max_clearance_bias", 0.08);
    declare_parameter("one_side_center_offset", 0.38);
    declare_parameter("rear_axle_lookahead", 0.90);
    declare_parameter("steering_gain", 2.60);
    declare_parameter("target_filter_alpha", 0.55);
    declare_parameter("steering_filter_alpha", 0.55);
    declare_parameter("target_reversal_alpha", 0.85);
    declare_parameter("steering_reversal_alpha", 0.85);
    declare_parameter("steering_reversal_delta_deg", 10.0);
    declare_parameter("steering_reversal_min_angle_deg", 12.0);
    declare_parameter("steering_reversal_memory_scans", 10);
    declare_parameter("s_turn_hold_scans", 7);
    declare_parameter("stale_scan_timeout", 0.30);

    debug_mode_ = get_parameter("debug_mode").as_bool();
    lidar_yaw_ = static_cast<float>(get_parameter("lidar_yaw_deg").as_double() * PI / 180.0);
    invert_scan_angle_ = get_parameter("invert_scan_angle").as_bool();
    scan_half_angle_ = static_cast<float>(get_parameter("scan_half_angle_deg").as_double() * PI / 180.0);
    cone_min_dist_ = static_cast<float>(get_parameter("cone_min_dist").as_double());
    cone_max_dist_ = static_cast<float>(get_parameter("cone_max_dist").as_double());
    max_x_limit_ = static_cast<float>(get_parameter("max_x_limit").as_double());
    min_cluster_points_ = static_cast<int>(get_parameter("min_cluster_points").as_int());
    min_cluster_span_ = static_cast<float>(get_parameter("min_cluster_span").as_double());
    cone_base_radius_ =
        static_cast<float>(get_parameter("cone_base_radius").as_double());
    cone_diameter_tolerance_ =
        static_cast<float>(get_parameter("cone_diameter_tolerance").as_double());
    cone_radius_fit_tolerance_ =
        static_cast<float>(get_parameter("cone_radius_fit_tolerance").as_double());
    max_cluster_span_ = static_cast<float>(get_parameter("max_cluster_span").as_double());
    max_cone_shape_span_ =
        static_cast<float>(get_parameter("max_cone_shape_span").as_double());
    cone_base_radius_ = std::max(0.01f, cone_base_radius_);
    cone_diameter_tolerance_ = std::max(0.0f, cone_diameter_tolerance_);
    cone_radius_fit_tolerance_ = std::max(0.0f, cone_radius_fit_tolerance_);
    max_cluster_span_ = std::min(
        max_cluster_span_, 2.0f * cone_base_radius_ + cone_diameter_tolerance_);
    max_cone_shape_span_ =
        std::clamp(max_cone_shape_span_, min_cluster_span_, max_cluster_span_);
    max_linearity_ = static_cast<float>(get_parameter("max_linearity").as_double());
    min_sagitta_ = static_cast<float>(get_parameter("min_sagitta").as_double());
    wall_fragment_seed_distance_ = static_cast<float>(
        get_parameter("wall_fragment_seed_distance").as_double());
    wall_fragment_expand_distance_ = static_cast<float>(
        get_parameter("wall_fragment_expand_distance").as_double());
    wall_fragment_match_distance_ = static_cast<float>(
        get_parameter("wall_fragment_match_distance").as_double());
    wall_fragment_min_group_ = static_cast<int>(
        get_parameter("wall_fragment_min_group").as_int());
    wall_fragment_memory_scans_ = static_cast<int>(
        get_parameter("wall_fragment_memory_scans").as_int());
    wall_fragment_seed_distance_ = std::max(0.05f, wall_fragment_seed_distance_);
    wall_fragment_expand_distance_ = std::max(
        wall_fragment_seed_distance_, wall_fragment_expand_distance_);
    wall_fragment_match_distance_ = std::max(
        0.05f, wall_fragment_match_distance_);
    wall_fragment_min_group_ = std::max(3, wall_fragment_min_group_);
    wall_fragment_memory_scans_ = std::max(0, wall_fragment_memory_scans_);
    drive_speed_ = static_cast<float>(get_parameter("drive_speed").as_double());
    one_side_speed_ = static_cast<float>(get_parameter("one_side_speed").as_double());
    stable_one_side_speed_ =
        static_cast<float>(get_parameter("stable_one_side_speed").as_double());
    stable_one_side_scans_ =
        static_cast<int>(get_parameter("stable_one_side_scans").as_int());
    curve_speed_ = static_cast<float>(get_parameter("curve_speed").as_double());
    reversal_speed_ = static_cast<float>(get_parameter("reversal_speed").as_double());
    clearance_speed_ = static_cast<float>(get_parameter("clearance_speed").as_double());
    full_speed_angle_deg_ =
        static_cast<float>(get_parameter("full_speed_angle_deg").as_double());
    slow_speed_angle_deg_ =
        static_cast<float>(get_parameter("slow_speed_angle_deg").as_double());
    recovery_speed_ = static_cast<float>(get_parameter("recovery_speed").as_double());
    singleton_hold_scans_ = static_cast<int>(get_parameter("singleton_hold_scans").as_int());
    empty_hold_scans_ = static_cast<int>(get_parameter("empty_hold_scans").as_int());
    speed_accel_step_ = static_cast<float>(get_parameter("speed_accel_step").as_double());
    speed_decel_step_ = static_cast<float>(get_parameter("speed_decel_step").as_double());
    drive_speed_ = std::max(0.0f, drive_speed_);
    curve_speed_ = std::clamp(curve_speed_, 0.0f, drive_speed_);
    one_side_speed_ = std::clamp(one_side_speed_, 0.0f, drive_speed_);
    stable_one_side_speed_ =
        std::clamp(stable_one_side_speed_, one_side_speed_, drive_speed_);
    stable_one_side_scans_ = std::max(1, stable_one_side_scans_);
    reversal_speed_ = std::clamp(reversal_speed_, 0.0f, curve_speed_);
    clearance_speed_ = std::clamp(clearance_speed_, 0.0f, curve_speed_);
    full_speed_angle_deg_ = std::max(0.0f, full_speed_angle_deg_);
    slow_speed_angle_deg_ = std::max(full_speed_angle_deg_ + 1.0f, slow_speed_angle_deg_);
    recovery_speed_ = std::clamp(recovery_speed_, 0.0f, curve_speed_);
    singleton_hold_scans_ = std::max(0, singleton_hold_scans_);
    empty_hold_scans_ = std::max(0, empty_hold_scans_);
    speed_accel_step_ = std::max(0.01f, speed_accel_step_);
    speed_decel_step_ = std::max(0.01f, speed_decel_step_);
    min_lane_width_ = static_cast<float>(get_parameter("min_lane_width").as_double());
    max_lane_width_ = static_cast<float>(get_parameter("max_lane_width").as_double());
    vehicle_length_ = static_cast<float>(get_parameter("vehicle_length").as_double());
    vehicle_width_ = static_cast<float>(get_parameter("vehicle_width").as_double());
    wheelbase_ = static_cast<float>(get_parameter("wheelbase").as_double());
    rear_overhang_ = static_cast<float>(get_parameter("rear_overhang").as_double());
    front_overhang_ = static_cast<float>(get_parameter("front_overhang").as_double());
    lidar_ahead_of_front_axle_ =
        static_cast<float>(get_parameter("lidar_ahead_of_front_axle").as_double());
    body_clearance_margin_ =
        static_cast<float>(get_parameter("body_clearance_margin").as_double());
    near_cone_check_forward_ =
        static_cast<float>(get_parameter("near_cone_check_forward").as_double());
    max_clearance_bias_ =
        static_cast<float>(get_parameter("max_clearance_bias").as_double());
    one_side_center_offset_ =
        static_cast<float>(get_parameter("one_side_center_offset").as_double());
    rear_axle_lookahead_ =
        static_cast<float>(get_parameter("rear_axle_lookahead").as_double());
    steering_gain_ = static_cast<float>(get_parameter("steering_gain").as_double());
    target_filter_alpha_ =
        static_cast<float>(get_parameter("target_filter_alpha").as_double());
    steering_filter_alpha_ =
        static_cast<float>(get_parameter("steering_filter_alpha").as_double());
    target_reversal_alpha_ =
        static_cast<float>(get_parameter("target_reversal_alpha").as_double());
    steering_reversal_alpha_ =
        static_cast<float>(get_parameter("steering_reversal_alpha").as_double());
    steering_reversal_delta_deg_ =
        static_cast<float>(get_parameter("steering_reversal_delta_deg").as_double());
    steering_reversal_min_angle_deg_ = static_cast<float>(
        get_parameter("steering_reversal_min_angle_deg").as_double());
    steering_reversal_memory_scans_ =
        static_cast<int>(get_parameter("steering_reversal_memory_scans").as_int());
    s_turn_hold_scans_ = static_cast<int>(get_parameter("s_turn_hold_scans").as_int());
    target_filter_alpha_ = std::clamp(target_filter_alpha_, 0.05f, 1.0f);
    steering_filter_alpha_ = std::clamp(steering_filter_alpha_, 0.05f, 1.0f);
    target_reversal_alpha_ =
        std::clamp(target_reversal_alpha_, target_filter_alpha_, 1.0f);
    steering_reversal_alpha_ =
        std::clamp(steering_reversal_alpha_, steering_filter_alpha_, 1.0f);
    steering_reversal_delta_deg_ = std::max(1.0f, steering_reversal_delta_deg_);
    steering_reversal_min_angle_deg_ =
        std::max(1.0f, steering_reversal_min_angle_deg_);
    steering_reversal_memory_scans_ = std::max(1, steering_reversal_memory_scans_);
    s_turn_hold_scans_ = std::max(0, s_turn_hold_scans_);
    near_cone_check_forward_ = std::max(0.05f, near_cone_check_forward_);
    max_clearance_bias_ = std::clamp(max_clearance_bias_, 0.0f, 0.10f);
    one_side_center_offset_ = std::clamp(one_side_center_offset_, 0.30f, 0.45f);
    lidar_to_rear_axle_ = wheelbase_ + lidar_ahead_of_front_axle_;
    stale_scan_timeout_ = static_cast<float>(get_parameter("stale_scan_timeout").as_double());

    const float measured_length = rear_overhang_ + wheelbase_ + front_overhang_;
    if (std::abs(measured_length - vehicle_length_) > 0.05f) {
      RCLCPP_WARN(get_logger(),
                  "Vehicle dimensions disagree: length=%.2f m but overhang+wheelbase=%.2f m",
                  vehicle_length_, measured_length);
    }
    const float rear_safety_radius =
        std::hypot(vehicle_width_ * 0.5f, rear_overhang_) + body_clearance_margin_;
    if (rear_safety_radius >= min_lane_width_ * 0.5f) {
      RCLCPP_WARN(get_logger(),
                  "Rear safety envelope %.2f m is too large for minimum lane half-width %.2f m",
                  rear_safety_radius, min_lane_width_ * 0.5f);
    }

    motor_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/xycar_motor", 1);
    lane_status_pub_ = create_publisher<std_msgs::msg::UInt8>(
        "/final_drive/cone_lane_status", 1);
    auto qos = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort();
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", qos, std::bind(&ConeDriveV2::scanCallback, this, std::placeholders::_1));
    motor_timer_ = create_wall_timer(30ms, std::bind(&ConeDriveV2::publishLatestCommand, this));

    RCLCPP_INFO(get_logger(),
                "ConeDriveV2: lidar_yaw=%.1f deg, invert_angle=%s, vehicle FOV=+/-%.1f deg, min_points=%d",
                lidar_yaw_ * 180.0f / PI, invert_scan_angle_ ? "true" : "false",
                scan_half_angle_ * 180.0f / PI, min_cluster_points_);
    RCLCPP_INFO(get_logger(),
                "Vehicle geometry: %.2f x %.2f m, wheelbase=%.2f m, LiDAR is %.2f m ahead of rear axle, rear safety radius=%.2f m",
                vehicle_length_, vehicle_width_, wheelbase_, lidar_to_rear_axle_,
                rear_safety_radius);
    RCLCPP_INFO(get_logger(),
                "Lane-loss recovery: singleton=%d scans, empty=%d scans, creep=%.1f, speed steps=+%.2f/-%.2f",
                singleton_hold_scans_, empty_hold_scans_, recovery_speed_,
                speed_accel_step_, speed_decel_step_);
    RCLCPP_INFO(get_logger(),
                "Speed profile: straight=%.1f through %.1f deg, one-side=%.1f -> %.1f after %d scans, curve=%.1f from %.1f deg",
                drive_speed_, full_speed_angle_deg_, one_side_speed_,
                stable_one_side_speed_, stable_one_side_scans_, curve_speed_,
                slow_speed_angle_deg_);
    RCLCPP_INFO(get_logger(),
                "Steering preview: rear-axle lookahead=%.2f m, gain=%.2f, target alpha=%.2f/%.2f, steering alpha=%.2f/%.2f, S-hold=%d",
                rear_axle_lookahead_, steering_gain_, target_filter_alpha_,
                target_reversal_alpha_, steering_filter_alpha_,
                steering_reversal_alpha_, s_turn_hold_scans_);
  }

  ~ConeDriveV2() override { publishMotor(0.0f, 0.0f); }

private:
  std::vector<std::vector<ScanPoint>> groupScan(const sensor_msgs::msg::LaserScan &scan) const {
    std::vector<std::vector<ScanPoint>> groups;
    std::vector<ScanPoint> current;
    int invalid_run = 0;

    const auto noteInvalidPoint = [&]() {
      invalid_run++;
      // Preserve a cone across one or two missing returns, but never bridge a
      // wide empty angular sector or the excluded rear half-plane.
      if (invalid_run > 2 && !current.empty()) {
        groups.push_back(std::move(current));
        current.clear();
      }
    };

    for (size_t i = 0; i < scan.ranges.size(); ++i) {
      const float range = scan.ranges[i];
      const float measured_angle = scan.angle_min + static_cast<float>(i) * scan.angle_increment;
      const float raw_angle = invert_scan_angle_ ? -measured_angle : measured_angle;
      const float angle = normalizeAngle(raw_angle + lidar_yaw_);

      const bool valid = std::isfinite(range) && range >= cone_min_dist_ &&
                         range <= cone_max_dist_ && std::abs(angle) <= scan_half_angle_;
      if (!valid) {
        noteInvalidPoint();
        continue;
      }

      ScanPoint point;
      point.index = static_cast<int>(i);
      point.angle = angle;
      point.range = range;
      point.x = range * std::sin(angle);
      point.y = range * std::cos(angle);
      if (std::abs(point.x) > max_x_limit_ || point.y < 0.0f) {
        noteInvalidPoint();
        continue;
      }

      invalid_run = 0;
      if (!current.empty()) {
        const float threshold = 0.035f + range * 0.010f;
        if (point.distanceTo(current.back()) > threshold) {
          groups.push_back(std::move(current));
          current.clear();
        }
      }
      current.push_back(point);
    }
    if (!current.empty()) groups.push_back(std::move(current));

    // With a 180-degree mount, vehicle-forward may lie at both ends of the
    // LaserScan array. Rejoin a cone split across index 0 / index N-1.
    if (groups.size() >= 2) {
      auto &first = groups.front();
      auto &last = groups.back();
      const bool touches_start = first.front().index <= 2;
      const bool touches_end = last.back().index >= static_cast<int>(scan.ranges.size()) - 3;
      if (touches_start && touches_end && first.front().distanceTo(last.back()) < 0.07f) {
        std::vector<ScanPoint> joined;
        joined.reserve(first.size() + last.size());
        joined.insert(joined.end(), last.begin(), last.end());
        joined.insert(joined.end(), first.begin(), first.end());
        groups.front() = std::move(joined);
        groups.pop_back();
      }
    }
    return groups;
  }

  bool makeCone(const std::vector<ScanPoint> &group, ConeCluster &cone) const {
    if (static_cast<int>(group.size()) < min_cluster_points_) return false;

    float sum_x = 0.0f, sum_y = 0.0f;
    for (const auto &point : group) {
      sum_x += point.x;
      sum_y += point.y;
    }
    cone.x = sum_x / static_cast<float>(group.size());
    cone.y = sum_y / static_cast<float>(group.size());
    cone.point_count = static_cast<int>(group.size());
    // Use the largest measured point-to-point distance as the observed
    // diameter.  The previous bounding-box diagonal could overstate a cone's
    // size, while its 28 cm limit allowed the 18--19 cm cylinder in frame 46.
    cone.span = 0.0f;
    for (size_t i = 0; i < group.size(); ++i) {
      for (size_t j = i + 1; j < group.size(); ++j) {
        cone.span = std::max(cone.span, group[i].distanceTo(group[j]));
        if (cone.span > max_cluster_span_) return false;
      }
    }
    if (cone.span < min_cluster_span_ || cone.span > max_cluster_span_) return false;

    float c_xx = 0.0f, c_yy = 0.0f, c_xy = 0.0f;
    for (const auto &point : group) {
      const float dx = point.x - cone.x;
      const float dy = point.y - cone.y;
      c_xx += dx * dx;
      c_yy += dy * dy;
      c_xy += dx * dy;
    }
    const float n = static_cast<float>(group.size());
    c_xx /= n;
    c_yy /= n;
    c_xy /= n;
    const float trace = c_xx + c_yy;
    const float determinant = c_xx * c_yy - c_xy * c_xy;
    const float gap = std::sqrt(std::max(0.0f, trace * trace * 0.25f - determinant));
    const float lambda1 = trace * 0.5f + gap;
    const float lambda2 = std::max(1e-6f, trace * 0.5f - gap);
    cone.linearity = std::sqrt(lambda1 / lambda2);

    const auto &first = group.front();
    const auto &last = group.back();
    const float chord_x = last.x - first.x;
    const float chord_y = last.y - first.y;
    const float chord = std::hypot(chord_x, chord_y);
    cone.sagitta = 0.0f;
    float signed_peak_depth = 0.0f;
    if (chord > 0.025f) {
      for (size_t i = 1; i + 1 < group.size(); ++i) {
        const float signed_depth =
            (chord_x * (group[i].y - first.y) -
             chord_y * (group[i].x - first.x)) / chord;
        const float depth = std::abs(signed_depth);
        if (depth > cone.sagitta) {
          cone.sagitta = depth;
          signed_peak_depth = signed_depth;
        }
      }
    }
    if (chord > 0.03f && cone.sagitta > 0.003f) {
      const float sensor_side =
          (chord_x * -first.y - chord_y * -first.x) / chord;
      cone.faces_sensor = sensor_side * signed_peak_depth >= 0.0f;
    }

    // A partially visible large cylinder can have a short enough chord to
    // pass the diameter limit.  Estimate the arc radius from chord/sagitta and
    // reject it when it is clearly larger than the 7 cm cone base radius.
    if (chord > 0.10f && cone.sagitta > min_sagitta_) {
      const float fitted_radius =
          chord * chord / (8.0f * cone.sagitta) + cone.sagitta * 0.5f;
      if (fitted_radius > cone_base_radius_ + cone_radius_fit_tolerance_) return false;
    }

    // A real cone is a compact multi-point arc. Long straight chair/wall edges
    // fail the linearity/curvature test. Short arcs do not need a large sagitta.
    if (cone.span > 0.07f && cone.linearity > max_linearity_) return false;
    if (cone.span > 0.09f && cone.sagitta < min_sagitta_) return false;
    return true;
  }

  std::vector<ConeCluster> detectCones(const sensor_msgs::msg::LaserScan &scan) const {
    std::vector<ConeCluster> cones;
    for (const auto &group : groupScan(scan)) {
      ConeCluster cone;
      if (makeCone(group, cone)) cones.push_back(cone);
    }
    return cones;
  }

  std::vector<ConeCluster> filterWallFragments(
      const std::vector<ConeCluster> &candidates) {
    const size_t count = candidates.size();
    std::vector<std::vector<size_t>> neighbors(count);
    for (size_t i = 0; i < count; ++i) {
      for (size_t j = i + 1; j < count; ++j) {
        if (candidates[i].distanceTo(candidates[j]) <
            wall_fragment_seed_distance_) {
          neighbors[i].push_back(j);
          neighbors[j].push_back(i);
        }
      }
    }

    // Rough walls are split into several compact, cone-like arcs.  Separate
    // cones cannot have centres this close (course spacing is 25--35 cm), so a
    // connected group of three or more short-spaced clusters is a wall seed.
    std::vector<bool> wall(count, false);
    std::vector<bool> visited(count, false);
    for (size_t seed = 0; seed < count; ++seed) {
      if (visited[seed]) continue;
      std::vector<size_t> stack{seed};
      std::vector<size_t> component;
      visited[seed] = true;
      while (!stack.empty()) {
        const size_t current = stack.back();
        stack.pop_back();
        component.push_back(current);
        for (const size_t next : neighbors[current]) {
          if (visited[next]) continue;
          visited[next] = true;
          stack.push_back(next);
        }
      }
      if (static_cast<int>(component.size()) >= wall_fragment_min_group_) {
        for (const size_t index : component) wall[index] = true;
      }
    }

    // Include gaps at the ends of the same rough surface without reaching the
    // minimum real cone spacing of 25 cm.
    bool expanded = true;
    while (expanded) {
      expanded = false;
      for (size_t i = 0; i < count; ++i) {
        if (wall[i]) continue;
        for (size_t j = 0; j < count; ++j) {
          if (!wall[j]) continue;
          if (candidates[i].distanceTo(candidates[j]) <
              wall_fragment_expand_distance_) {
            wall[i] = true;
            expanded = true;
            break;
          }
        }
      }
    }

    // Remember a wall for a few scans.  Its fragmentation changes as the car
    // moves, so an individual bump may temporarily appear as a lone cone even
    // though the dense wall group was visible one frame earlier.  A proximity
    // match alone is deliberately bounded: in real_track_4 a valid 9-point,
    // 6.1 cm sensor-facing cone arc was repeatedly matched to an old wall
    // track, and the old code reset the memory on every match forever.
    const size_t previous_track_count = wall_fragment_tracks_.size();
    std::vector<bool> candidate_matched(count, false);
    std::vector<WallFragmentTrack> next_tracks;
    next_tracks.reserve(previous_track_count + count);
    for (size_t track_index = 0; track_index < previous_track_count;
         ++track_index) {
      const auto &track = wall_fragment_tracks_[track_index];
      int best = -1;
      float best_distance = wall_fragment_match_distance_;
      for (size_t i = 0; i < count; ++i) {
        if (candidate_matched[i]) continue;
        const float distance = candidates[i].distanceTo(track);
        if (distance < best_distance) {
          best_distance = distance;
          best = static_cast<int>(i);
        }
      }
      if (best >= 0) {
        const auto &candidate = candidates[static_cast<size_t>(best)];
        WallFragmentTrack updated;
        updated.x = 0.7f * candidate.x + 0.3f * track.x;
        updated.y = 0.7f * candidate.y + 0.3f * track.y;
        const bool supported_by_dense_wall = wall[static_cast<size_t>(best)];
        updated.unsupported_scans = supported_by_dense_wall
            ? 0 : track.unsupported_scans + 1;
        if (updated.unsupported_scans <= wall_fragment_memory_scans_) {
          next_tracks.push_back(updated);
          candidate_matched[static_cast<size_t>(best)] = true;
          wall[static_cast<size_t>(best)] = true;
        }
      } else if (track.unsupported_scans < wall_fragment_memory_scans_) {
        auto retained = track;
        retained.unsupported_scans++;
        next_tracks.push_back(retained);
      }
    }
    for (size_t i = 0; i < count; ++i) {
      if (!wall[i] || candidate_matched[i]) continue;
      const bool already_tracked = std::any_of(
          next_tracks.begin(), next_tracks.end(), [&](const auto &track) {
            return candidates[i].distanceTo(track) < 0.08f;
          });
      if (already_tracked) continue;
      WallFragmentTrack added;
      added.x = candidates[i].x;
      added.y = candidates[i].y;
      added.unsupported_scans = 0;
      next_tracks.push_back(added);
    }
    wall_fragment_tracks_ = std::move(next_tracks);

    std::vector<ConeCluster> cones;
    cones.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      if (wall[i]) continue;
      // The visible chord of every confirmed cone in the real-track bags is
      // below 10 cm. Larger arcs and surfaces curving away from the LiDAR are
      // wall/cylinder fragments, not the sensor-facing side of a cone.
      if (candidates[i].span > max_cone_shape_span_) continue;
      if (!candidates[i].faces_sensor) continue;
      cones.push_back(candidates[i]);
    }
    return cones;
  }

  std::vector<Point> updateTracks(const std::vector<ConeCluster> &cones) {
    std::vector<bool> matched(tracks_.size(), false);
    for (const auto &cone : cones) {
      int best = -1;
      float best_distance = 0.16f;
      for (size_t i = 0; i < tracks_.size(); ++i) {
        if (matched[i]) continue;
        const float distance = cone.distanceTo(tracks_[i]);
        if (distance < best_distance) {
          best_distance = distance;
          best = static_cast<int>(i);
        }
      }
      if (best >= 0) {
        auto &track = tracks_[best];
        track.x = 0.7f * cone.x + 0.3f * track.x;
        track.y = 0.7f * cone.y + 0.3f * track.y;
        track.hits++;
        track.missing = 0;
        matched[best] = true;
      } else {
        TrackedCone track;
        track.x = cone.x;
        track.y = cone.y;
        tracks_.push_back(track);
        matched.push_back(true);
      }
    }

    std::vector<TrackedCone> retained;
    for (size_t i = 0; i < tracks_.size(); ++i) {
      if (i >= matched.size() || !matched[i]) tracks_[i].missing++;
      if (tracks_[i].missing <= 2) retained.push_back(tracks_[i]);
    }
    tracks_ = std::move(retained);

    std::vector<Point> confirmed;
    for (const auto &track : tracks_) {
      if (track.hits >= 2 && track.missing == 0) confirmed.push_back({track.x, track.y});
    }
    return confirmed;
  }

  std::vector<Point> makeLane(const std::vector<Point> &points, int start,
                              std::vector<bool> &used) const {
    std::vector<Point> lane{points[start]};
    used[start] = true;
    while (true) {
      int best = -1;
      float best_score = std::numeric_limits<float>::max();
      Point direction{0.0f, 1.0f};
      if (lane.size() >= 2) {
        const float length = lane.back().distanceTo(lane[lane.size() - 2]);
        direction = {(lane.back().x - lane[lane.size() - 2].x) / length,
                     (lane.back().y - lane[lane.size() - 2].y) / length};
      }
      for (size_t i = 0; i < points.size(); ++i) {
        if (used[i]) continue;
        const float distance = lane.back().distanceTo(points[i]);
        const bool normal_step = distance >= 0.18f && distance <= 0.42f;
        const bool one_missing = distance > 0.42f && distance <= 0.72f;
        if (!normal_step && !one_missing) continue;
        const Point step{(points[i].x - lane.back().x) / distance,
                         (points[i].y - lane.back().y) / distance};
        if (step.y < -0.10f) continue;
        const float turn = std::acos(std::clamp(direction.x * step.x + direction.y * step.y,
                                                -1.0f, 1.0f)) * 180.0f / PI;
        // The first edge of a sharply curving row is compared with the
        // vehicle-forward axis because a one-point lane has no measured
        // tangent yet.  real_track_4 frames 76..80 contained a valid 0.42 m
        // same-row edge at about 54 deg; the old 45 deg gate left both rows as
        // singletons and the lane-loss safety logic correctly stopped the car.
        // A 0.48 m distance cap keeps this relaxed first-edge rule far below
        // the 0.70 m minimum track width, so it cannot bridge to the opposite
        // cone row.  Later edges retain the stricter continuity gate.
        const bool sharp_first_edge = lane.size() == 1 && distance <= 0.48f;
        const float max_turn = sharp_first_edge ? 70.0f :
                               (one_missing ? 45.0f : 48.0f);
        if (turn > max_turn) continue;
        const float expected = one_missing ? 2.0f * CONE_SPACING_M : CONE_SPACING_M;
        const float score = turn + std::abs(distance - expected) * 60.0f +
                            (one_missing ? 25.0f : 0.0f);
        if (score < best_score) {
          best_score = score;
          best = static_cast<int>(i);
        }
      }
      if (best < 0) break;
      used[best] = true;
      lane.push_back(points[best]);
    }
    return lane;
  }

  int findHistorySeed(const std::vector<Point> &points, const std::vector<Point> &history,
                      const std::vector<bool> &used) const {
    int best = -1;
    float best_score = std::numeric_limits<float>::max();
    for (size_t i = 0; i < points.size(); ++i) {
      if (used[i]) continue;
      float history_distance = std::numeric_limits<float>::max();
      for (const auto &old : history) {
        history_distance = std::min(history_distance, points[i].distanceTo(old));
      }
      if (history_distance > 0.36f) continue;
      const float score = points[i].y + 0.35f * history_distance;
      if (score < best_score) {
        best = static_cast<int>(i);
        best_score = score;
      }
    }
    return best;
  }

  int findDefaultSeed(const std::vector<Point> &points, bool is_left,
                      const std::vector<bool> &used) const {
    const Point pivot{is_left ? HALF_WIDTH_M : -HALF_WIDTH_M, CONE_SPACING_M};
    int best = -1;
    float best_distance = 0.65f;
    for (size_t i = 0; i < points.size(); ++i) {
      if (used[i]) continue;
      if (is_left && points[i].x < -0.10f) continue;
      if (!is_left && points[i].x > 0.10f) continue;
      const float distance = points[i].distanceTo(pivot);
      if (distance < best_distance) {
        best = static_cast<int>(i);
        best_distance = distance;
      }
    }
    return best;
  }

  float laneAssociation(const std::vector<Point> &lane,
                        const std::vector<Point> &history) const {
    if (lane.empty() || history.empty()) return std::numeric_limits<float>::max();
    float sum = 0.0f;
    for (const auto &point : lane) {
      float nearest = std::numeric_limits<float>::max();
      for (const auto &old : history) nearest = std::min(nearest, point.distanceTo(old));
      sum += nearest;
    }
    return sum / static_cast<float>(lane.size());
  }

  bool validLanePair(const std::vector<Point> &left,
                     const std::vector<Point> &right) const {
    if (left.size() < 2 || right.size() < 2) return false;
    for (const auto &lpoint : left) {
      for (const auto &rpoint : right) {
        if (std::abs(lpoint.y - rpoint.y) > 0.35f) continue;
        const float width = lpoint.distanceTo(rpoint);
        if (width >= min_lane_width_ && width <= max_lane_width_) return true;
      }
    }
    return false;
  }

  std::pair<std::vector<Point>, std::vector<Point>> buildLanes(std::vector<Point> points) {
    std::sort(points.begin(), points.end(), [](const Point &a, const Point &b) {
      return a.x * a.x + a.y * a.y < b.x * b.x + b.y * b.y;
    });
    std::vector<bool> used(points.size(), false);
    std::vector<Point> left;
    std::vector<Point> right;

    const auto build_side = [&](bool is_left) {
      auto &lane = is_left ? left : right;
      const auto &history = is_left ? previous_left_ : previous_right_;
      int seed = findHistorySeed(points, history, used);
      // Do not jump to the opposite side while a recent side history exists.
      if (seed < 0 && history.empty()) seed = findDefaultSeed(points, is_left, used);
      if (seed >= 0) lane = makeLane(points, seed, used);
    };

    // The side with the longer previous chain claims its continuation first.
    // This is what preserves the right row through frames 120..140 even after
    // that row crosses x=0 during the sharp left turn.
    if (previous_right_.size() > previous_left_.size()) {
      build_side(false);
      build_side(true);
    } else {
      build_side(true);
      build_side(false);
    }

    if (!left.empty() && !right.empty() && !validLanePair(left, right)) {
      bool keep_left = false;
      if (left.size() != right.size()) {
        keep_left = left.size() > right.size();
      } else {
        keep_left = laneAssociation(left, previous_left_) <
                    laneAssociation(right, previous_right_);
      }
      if (keep_left) right.clear();
      else left.clear();
    }

    const auto update_history = [](const std::vector<Point> &lane,
                                   std::vector<Point> &history, int &missing) {
      if (lane.size() >= 2) {
        history = lane;
        missing = 0;
      } else {
        missing++;
        if (missing > 4) history.clear();
      }
    };
    update_history(left, previous_left_, left_missing_frames_);
    update_history(right, previous_right_, right_missing_frames_);
    return {left, right};
  }

  std::vector<Point> offsetLaneToCenter(const std::vector<Point> &lane,
                                        bool is_left) const {
    std::vector<Point> center;
    center.reserve(lane.size());
    for (size_t i = 0; i < lane.size(); ++i) {
      const auto &before = lane[i == 0 ? 0 : i - 1];
      const auto &after = lane[std::min(i + 1, lane.size() - 1)];
      const float length = std::max(1e-4f, before.distanceTo(after));
      const Point tangent{(after.x - before.x) / length, (after.y - before.y) / length};
      const Point right_normal{-tangent.y, tangent.x};
      const float sign = is_left ? 1.0f : -1.0f;
      center.push_back({lane[i].x + sign * one_side_center_offset_ * right_normal.x,
                        lane[i].y + sign * one_side_center_offset_ * right_normal.y});
    }
    return center;
  }

  std::pair<float, float> laneLossCommand(bool has_singleton_hint) {
    // A missing frame breaks one-side continuity.  Reacquisition must prove
    // the same side for several scans before the higher stable cap is allowed.
    last_one_side_id_ = 0;
    consecutive_one_side_scans_ = 0;
    lane_lost_scans_++;
    const int hold_scans = has_singleton_hint ? singleton_hold_scans_ : empty_hold_scans_;
    if (have_reliable_lane_ && lane_lost_scans_ <= hold_scans) {
      control_mode_ = "RECOVERY";
      if (debug_mode_) return {latest_angle_, 0.0f};

      // A single missed 10 Hz scan used to force 3.0 -> 0.8 -> 3.0 and made
      // the vehicle visibly hesitate.  Keep only the first missed frame near
      // the previous command (never above the one-side cap), then slow down if
      // perception is still missing on the next scan.
      const float loss_speed = lane_lost_scans_ == 1
          ? std::clamp(latest_speed_, recovery_speed_, one_side_speed_)
          : recovery_speed_;
      return {latest_angle_, loss_speed};
    }

    // Do not keep reviving an old path after the short recovery window. A new
    // reliable 2+ cone lane must be acquired before creep is allowed again.
    control_mode_ = "STOP";
    have_reliable_lane_ = false;
    target_initialized_ = false;
    return {latest_angle_, 0.0f};
  }

  std::pair<float, float> calculateCommand(
      const std::vector<Point> &left, const std::vector<Point> &right,
      const std::vector<ConeCluster> &current_cones) {
    const bool reliable_left = left.size() >= 2;
    const bool reliable_right = right.size() >= 2;
    if (!reliable_left && !reliable_right) {
      return laneLossCommand(!left.empty() || !right.empty());
    }

    std::vector<Point> center_path;
    if (reliable_left && reliable_right) {
      for (const auto &lpoint : left) {
        const Point *best = nullptr;
        float best_y_gap = 0.35f;
        for (const auto &rpoint : right) {
          const float gap = std::abs(lpoint.y - rpoint.y);
          if (gap < best_y_gap) {
            best_y_gap = gap;
            best = &rpoint;
          }
        }
        if (best != nullptr) {
          center_path.push_back({(lpoint.x + best->x) * 0.5f,
                                 (lpoint.y + best->y) * 0.5f});
        }
      }
    } else if (reliable_left) {
      center_path = offsetLaneToCenter(left, true);
    } else {
      center_path = offsetLaneToCenter(right, false);
    }
    if (center_path.empty()) return laneLossCommand(true);

    lane_lost_scans_ = 0;
    have_reliable_lane_ = true;
    control_mode_ = "TRACK";

    // A single visible boundary is normal in a sharp corner, but the first
    // couple of frames can also be a transient lane-identity change.  Keep the
    // conservative cap during that transition and allow the faster cap only
    // after the same physical side has remained reliable for several scans.
    const int one_side_id = reliable_left == reliable_right
        ? 0 : (reliable_left ? 1 : -1);
    if (one_side_id == 0) {
      last_one_side_id_ = 0;
      consecutive_one_side_scans_ = 0;
    } else if (one_side_id == last_one_side_id_) {
      consecutive_one_side_scans_++;
    } else {
      last_one_side_id_ = one_side_id;
      consecutive_one_side_scans_ = 1;
    }

    const auto target_it = std::min_element(
        center_path.begin(), center_path.end(), [this](const Point &a, const Point &b) {
          const float a_rear_y = a.y + lidar_to_rear_axle_;
          const float b_rear_y = b.y + lidar_to_rear_axle_;
          return std::abs(a_rear_y - rear_axle_lookahead_) <
                 std::abs(b_rear_y - rear_axle_lookahead_);
        });
    Point target = *target_it;
    bool target_reversing = false;
    if (!target_initialized_) {
      filtered_target_x_ = target.x;
      target_initialized_ = true;
    } else {
      target_reversing =
          filtered_target_x_ * target.x < 0.0f &&
          std::abs(target.x - filtered_target_x_) > 0.06f;
      const float target_alpha =
          target_reversing ? target_reversal_alpha_ : target_filter_alpha_;
      filtered_target_x_ = (1.0f - target_alpha) * filtered_target_x_ +
                           target_alpha * target.x;
    }
    target.x = filtered_target_x_;

    // A cone can pass only a few centimetres from the body while the center
    // path still looks correct.  In track_speed_1 frame 73, the nearest right
    // cone was at x=-0.228 m: its 7 cm base reached the 15 cm vehicle side.
    // Shift the immediate steering target away from the closest confirmed cone
    // until the body + cone radius + requested margin fits between them.
    const float required_cone_center_clearance =
        vehicle_width_ * 0.5f + cone_base_radius_ + body_clearance_margin_;
    float clearance_bias = 0.0f;
    float worst_clearance_deficit = 0.0f;
    const auto consider_clearance = [&](const std::vector<Point> &lane) {
      for (const auto &point : lane) {
        if (point.y < -0.05f || point.y > near_cone_check_forward_) continue;
        const float deficit = required_cone_center_clearance - std::abs(point.x);
        if (deficit <= worst_clearance_deficit) continue;
        worst_clearance_deficit = deficit;
        const float away_direction = point.x < 0.0f ? 1.0f : -1.0f;
        clearance_bias = away_direction * std::min(deficit, max_clearance_bias_);
      }
    };
    consider_clearance(left);
    consider_clearance(right);
    const bool clearance_limited = worst_clearance_deficit > 0.0f;
    // Tracking smooths lateral motion, which is useful for steering but can
    // delay a speed reduction by one scan.  Use current shape-validated cone
    // clusters as a speed-only early warning; they never alter lane identity or
    // steering.  This catches a rapidly approaching cone before its filtered
    // lane point enters the body-clearance envelope.
    const bool current_cone_clearance_limited =
        std::any_of(current_cones.begin(), current_cones.end(),
                    [&](const ConeCluster &cone) {
                      return cone.y >= -0.05f &&
                             cone.y <= near_cone_check_forward_ &&
                             std::abs(cone.x) < required_cone_center_clearance;
                    });
    target.x += clearance_bias;

    // All cone geometry is measured from the LiDAR, but steering rotates about
    // the rear axle. The LiDAR is 0.40 m ahead of that axle on this vehicle.
    // Using the rear-axle target prevents the back of the car from cutting
    // inside the cone line while the front-mounted sensor follows the center.
    const float lookahead = std::max(0.25f, target.y + lidar_to_rear_axle_);
    const float heading_error = std::atan2(-target.x, lookahead) * 180.0f / PI;
    const float raw_angle = steering_gain_ * heading_error;
    const float previous_angle = latest_angle_;
    const bool immediate_steering_reversal =
        previous_angle * raw_angle < 0.0f &&
        std::abs(previous_angle) >= steering_reversal_min_angle_deg_ &&
        std::abs(raw_angle) >= steering_reversal_min_angle_deg_ &&
        std::abs(raw_angle - previous_angle) >= steering_reversal_delta_deg_;
    const int raw_steering_sign =
        raw_angle >= steering_reversal_min_angle_deg_ ? 1 :
        raw_angle <= -steering_reversal_min_angle_deg_ ? -1 : 0;
    const bool remembered_steering_reversal =
        raw_steering_sign != 0 && last_strong_steering_sign_ != 0 &&
        raw_steering_sign != last_strong_steering_sign_ &&
        steering_reversal_memory_remaining_ > 0;
    const bool steering_reversing =
        immediate_steering_reversal || remembered_steering_reversal;
    const float steering_alpha =
        steering_reversing ? steering_reversal_alpha_ : steering_filter_alpha_;
    latest_angle_ = std::clamp(
        (1.0f - steering_alpha) * previous_angle + steering_alpha * raw_angle,
        -50.0f, 50.0f);
    if (raw_steering_sign != 0) {
      last_strong_steering_sign_ = raw_steering_sign;
      steering_reversal_memory_remaining_ = steering_reversal_memory_scans_;
    } else if (steering_reversal_memory_remaining_ > 0) {
      steering_reversal_memory_remaining_--;
    } else {
      last_strong_steering_sign_ = 0;
    }
    if (steering_reversing || target_reversing) {
      s_turn_slow_scans_remaining_ = s_turn_hold_scans_;
    } else if (s_turn_slow_scans_remaining_ > 0) {
      s_turn_slow_scans_remaining_--;
    }
    float speed = 0.0f;
    if (!debug_mode_) {
      // Use both the filtered command and the immediate steering demand for
      // speed control.  During an S-turn reversal the filtered angle crosses
      // zero even though a large opposite steering command is already needed;
      // using only latest_angle_ caused a short and unsafe re-acceleration.
      const float steering_demand =
          std::max(std::abs(latest_angle_), std::abs(raw_angle));
      if (steering_demand <= full_speed_angle_deg_) {
        speed = drive_speed_;
      } else if (steering_demand >= slow_speed_angle_deg_) {
        speed = curve_speed_;
      } else {
        const float ratio = (steering_demand - full_speed_angle_deg_) /
                            (slow_speed_angle_deg_ - full_speed_angle_deg_);
        speed = drive_speed_ + ratio * (curve_speed_ - drive_speed_);
      }
      if (!reliable_left || !reliable_right) {
        const bool stable_one_side =
            consecutive_one_side_scans_ >= stable_one_side_scans_;
        const float one_side_cap =
            stable_one_side ? stable_one_side_speed_ : one_side_speed_;
        speed = std::min(speed, one_side_cap);
      }
      if (s_turn_slow_scans_remaining_ > 0) speed = std::min(speed, reversal_speed_);
      if (clearance_limited || current_cone_clearance_limited) {
        speed = std::min(speed, clearance_speed_);
      }
    }
    return {latest_angle_, speed};
  }

  void drawDebug(const sensor_msgs::msg::LaserScan &scan,
                 const std::vector<ConeCluster> &cones,
                 const std::vector<Point> &left,
                 const std::vector<Point> &right) {
    if (!debug_mode_ || !debug_window_enabled_) return;
    constexpr int width = 700;
    constexpr int height = 620;
    constexpr int center_x = width / 2;
    constexpr int center_y = height - 55;
    constexpr float scale = 400.0f;
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(25, 25, 25));
    const auto pixel = [&](float x, float y) {
      // Vehicle-left (+x) is drawn on the left side of the screen.
      return cv::Point(center_x - static_cast<int>(x * scale),
                       center_y - static_cast<int>(y * scale));
    };

    for (float y = 0.3f; y <= 1.2f; y += 0.3f) {
      cv::line(canvas, pixel(-0.85f, y), pixel(0.85f, y), cv::Scalar(50, 50, 50), 1);
    }
    cv::line(canvas, pixel(0.0f, 0.0f), pixel(0.0f, 1.25f), cv::Scalar(65, 65, 65), 1);

    for (size_t i = 0; i < scan.ranges.size(); ++i) {
      const float range = scan.ranges[i];
      if (!std::isfinite(range) || range < 0.05f || range > 1.30f) continue;
      const float measured_angle = scan.angle_min + static_cast<float>(i) * scan.angle_increment;
      const float raw_angle = invert_scan_angle_ ? -measured_angle : measured_angle;
      const float angle = normalizeAngle(raw_angle + lidar_yaw_);
      if (std::abs(angle) > scan_half_angle_) continue;
      const float x = range * std::sin(angle);
      const float y = range * std::cos(angle);
      if (std::abs(x) > 0.85f || y < 0.0f) continue;
      cv::circle(canvas, pixel(x, y), 2, cv::Scalar(0, 125, 255), -1);
    }

    for (const auto &cone : cones) {
      const auto p = pixel(cone.x, cone.y);
      cv::circle(canvas, p, 9, cv::Scalar(0, 255, 80), 2);
      cv::putText(canvas, std::to_string(cone.point_count), p + cv::Point(7, -7),
                  cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(0, 255, 80), 1);
    }
    for (size_t i = 1; i < left.size(); ++i) {
      cv::line(canvas, pixel(left[i - 1].x, left[i - 1].y), pixel(left[i].x, left[i].y),
               cv::Scalar(0, 150, 255), 2);
    }
    for (const auto &point : left) cv::circle(canvas, pixel(point.x, point.y), 11, cv::Scalar(0, 150, 255), 2);
    for (size_t i = 1; i < right.size(); ++i) {
      cv::line(canvas, pixel(right[i - 1].x, right[i - 1].y), pixel(right[i].x, right[i].y),
               cv::Scalar(255, 150, 0), 2);
    }
    for (const auto &point : right) cv::circle(canvas, pixel(point.x, point.y), 11, cv::Scalar(255, 150, 0), 2);

    const bool left_ok = left.size() >= 2;
    const bool right_ok = right.size() >= 2;
    const std::string lane_status = left_ok && right_ok ? "BOTH" :
                                    left_ok ? "LEFT_ONLY" :
                                    right_ok ? "RIGHT_ONLY" : "NONE";

    cv::arrowedLine(canvas, pixel(0.0f, 0.0f), pixel(0.0f, 0.12f), cv::Scalar(230, 230, 230), 2);
    cv::putText(canvas, "LEFT (+x)", cv::Point(15, 25), cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(0, 150, 255), 1);
    cv::putText(canvas, "RIGHT (-x)", cv::Point(width - 125, 25), cv::FONT_HERSHEY_SIMPLEX,
                0.55, cv::Scalar(255, 150, 0), 1);
    cv::putText(canvas, "orange: scan  green: >=3pt cone candidate",
                cv::Point(15, height - 15), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(220, 220, 220), 1);
    cv::putText(canvas, "Lane: " + lane_status, cv::Point(15, 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(230, 230, 230), 1);
    try {
      cv::imshow("Cone Drive V2: vehicle -90..+90 deg", canvas);
      cv::waitKey(1);
    } catch (const cv::Exception &error) {
      debug_window_enabled_ = false;
      RCLCPP_WARN(get_logger(), "Debug window disabled: %s", error.what());
    }
  }

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan) {
    const int64_t stamp_ns = rclcpp::Time(scan->header.stamp).nanoseconds();
    if (stamp_ns > 0 && last_scan_stamp_ns_ >= 0 && stamp_ns <= last_scan_stamp_ns_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "Discarding duplicate/out-of-order /scan. Check that publisher count is 1.");
      return;
    }
    if (stamp_ns > 0) last_scan_stamp_ns_ = stamp_ns;

    // Perception/tracker state is advanced exactly once per new LaserScan.
    last_scan_time_ = now();
    const auto cone_candidates = detectCones(*scan);
    const auto cones = filterWallFragments(cone_candidates);
    const auto confirmed = updateTracks(cones);
    const auto [left, right] = buildLanes(confirmed);
    drawDebug(*scan, cones, left, right);
    const auto [angle, speed] = calculateCommand(left, right, cones);
    latest_angle_ = angle;
    latest_speed_ = speed;
    have_scan_ = true;

    const bool left_ok = left.size() >= 2;
    const bool right_ok = right.size() >= 2;
    std_msgs::msg::UInt8 lane_status;
    if (control_mode_ == "TRACK") {
      lane_status.data = left_ok && right_ok ? 2U : 1U;
    } else if (control_mode_ == "RECOVERY") {
      lane_status.data = 3U;
    } else {
      lane_status.data = 0U;
    }
    lane_status_pub_->publish(lane_status);
    const char *status = left_ok && right_ok ? "BOTH" :
                         left_ok ? "LEFT_ONLY" :
                         right_ok ? "RIGHT_ONLY" : "NONE";
    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 300,
        "[CONE V2] status=%s mode=%s candidates=%zu cones=%zu confirmed=%zu left=%zu right=%zu lost=%d angle=%+.1f target_speed=%.1f motor_speed=%.1f",
        status, control_mode_.c_str(), cone_candidates.size(), cones.size(),
        confirmed.size(), left.size(), right.size(), lane_lost_scans_,
        latest_angle_, latest_speed_, commanded_speed_);
  }

  float rampSpeed(float target) const {
    const float step = target > commanded_speed_ ? speed_accel_step_ : speed_decel_step_;
    if (std::abs(target - commanded_speed_) <= step) return target;
    return commanded_speed_ + std::copysign(step, target - commanded_speed_);
  }

  void publishLatestCommand() {
    if (!have_scan_) return;
    if ((now() - last_scan_time_).seconds() > stale_scan_timeout_) {
      // A stale sensor is different from a short lane-classification dropout:
      // stop immediately instead of applying the normal deceleration ramp.
      commanded_speed_ = 0.0f;
      publishMotor(latest_angle_, 0.0f);
      return;
    }
    commanded_speed_ = rampSpeed(latest_speed_);
    publishMotor(latest_angle_, commanded_speed_);
  }

  void publishMotor(float angle, float speed) {
    std_msgs::msg::Float32MultiArray command;
    command.data = {std::clamp(angle, -50.0f, 50.0f), speed};
    motor_pub_->publish(command);
  }

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr motor_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr lane_status_pub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::TimerBase::SharedPtr motor_timer_;

  bool debug_mode_{true};
  bool debug_window_enabled_{true};
  bool have_scan_{false};
  float lidar_yaw_{PI};
  bool invert_scan_angle_{false};
  float scan_half_angle_{PI * 0.5f};
  float cone_min_dist_{0.15f};
  float cone_max_dist_{1.20f};
  float max_x_limit_{0.85f};
  int min_cluster_points_{3};
  float min_cluster_span_{0.018f};
  float cone_base_radius_{0.07f};
  float cone_diameter_tolerance_{0.02f};
  float cone_radius_fit_tolerance_{0.02f};
  float max_cluster_span_{0.16f};
  float max_cone_shape_span_{0.10f};
  float max_linearity_{4.5f};
  float min_sagitta_{0.0015f};
  float wall_fragment_seed_distance_{0.16f};
  float wall_fragment_expand_distance_{0.22f};
  float wall_fragment_match_distance_{0.18f};
  int wall_fragment_min_group_{3};
  int wall_fragment_memory_scans_{4};
  float drive_speed_{9.5f};
  float one_side_speed_{6.0f};
  float stable_one_side_speed_{7.0f};
  int stable_one_side_scans_{3};
  float curve_speed_{7.5f};
  float reversal_speed_{5.5f};
  float clearance_speed_{5.5f};
  float full_speed_angle_deg_{8.0f};
  float slow_speed_angle_deg_{28.0f};
  float recovery_speed_{3.0f};
  int singleton_hold_scans_{5};
  int empty_hold_scans_{2};
  float speed_accel_step_{0.35f};
  float speed_decel_step_{0.45f};
  float min_lane_width_{0.70f};
  float max_lane_width_{0.98f};
  float vehicle_length_{0.60f};
  float vehicle_width_{0.30f};
  float wheelbase_{0.30f};
  float rear_overhang_{0.15f};
  float front_overhang_{0.15f};
  float lidar_ahead_of_front_axle_{0.10f};
  float lidar_to_rear_axle_{0.40f};
  float body_clearance_margin_{0.08f};
  float near_cone_check_forward_{0.30f};
  float max_clearance_bias_{0.08f};
  float one_side_center_offset_{0.38f};
  float rear_axle_lookahead_{0.90f};
  float steering_gain_{2.60f};
  float target_filter_alpha_{0.55f};
  float steering_filter_alpha_{0.55f};
  float target_reversal_alpha_{0.85f};
  float steering_reversal_alpha_{0.85f};
  float steering_reversal_delta_deg_{10.0f};
  float steering_reversal_min_angle_deg_{12.0f};
  int steering_reversal_memory_scans_{10};
  int steering_reversal_memory_remaining_{0};
  int last_strong_steering_sign_{0};
  int s_turn_hold_scans_{7};
  int s_turn_slow_scans_remaining_{0};
  int last_one_side_id_{0};
  int consecutive_one_side_scans_{0};
  float stale_scan_timeout_{0.30f};
  float latest_angle_{0.0f};
  float latest_speed_{0.0f};
  float commanded_speed_{0.0f};
  float filtered_target_x_{0.0f};
  bool target_initialized_{false};
  bool have_reliable_lane_{false};
  int lane_lost_scans_{0};
  std::string control_mode_{"STOP"};
  rclcpp::Time last_scan_time_{0, 0, RCL_ROS_TIME};
  int64_t last_scan_stamp_ns_{-1};
  std::vector<WallFragmentTrack> wall_fragment_tracks_;
  std::vector<TrackedCone> tracks_;
  std::vector<Point> previous_left_;
  std::vector<Point> previous_right_;
  int left_missing_frames_{0};
  int right_missing_frames_{0};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ConeDriveV2>());
  rclcpp::shutdown();
  cv::destroyAllWindows();
  return 0;
}
