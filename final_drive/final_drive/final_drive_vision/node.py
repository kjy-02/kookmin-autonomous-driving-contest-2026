"""ROS 2 traffic-light node with an optional checkerboard detector."""

from __future__ import annotations

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, UInt8

from .checkerboard_detector import CheckerboardDetector
from .detectors import (
    DetectionRoi,
    TemporalBoolFilter,
    TemporalStateFilter,
    TrafficLightDetector,
)
from .traffic_housing_detector import TrafficHousingDetector


class TrafficVisionNode(Node):
    def __init__(self) -> None:
        super().__init__("traffic_vision")
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("traffic_topic", "/traffic_light_state")
        self.declare_parameter("enable_checkerboard", False)
        self.declare_parameter("checkerboard_topic", "/checkerboard_detected")
        self.declare_parameter("checker_process_every_n_frames", 3)
        self.declare_parameter("traffic_filter_window", 5)
        self.declare_parameter("traffic_filter_hits", 3)
        self.declare_parameter("traffic_clear_hits", 2)
        self.declare_parameter("checker_on_hits", 2)
        self.declare_parameter("checker_off_hits", 2)
        self.declare_parameter("roi_left_fraction", 0.15)
        self.declare_parameter("roi_right_fraction", 0.15)
        self.declare_parameter("roi_top_fraction", 0.0)
        self.declare_parameter("roi_bottom_fraction", 0.15)
        self.declare_parameter("traffic_roi_max_y_fraction", 0.50)
        self.declare_parameter("traffic_model_path", "")
        self.declare_parameter("traffic_model_confidence", 0.35)
        self.declare_parameter("traffic_model_input_size", 640)
        self.declare_parameter("show_debug_window", False)

        image_topic = str(self.get_parameter("image_topic").value)
        traffic_topic = str(self.get_parameter("traffic_topic").value)
        checker_topic = str(self.get_parameter("checkerboard_topic").value)
        self._checker_enabled = bool(self.get_parameter("enable_checkerboard").value)
        self._checker_stride = max(
            1, int(self.get_parameter("checker_process_every_n_frames").value)
        )
        self._show_debug = bool(self.get_parameter("show_debug_window").value)

        self._detection_roi = DetectionRoi(
            left=float(self.get_parameter("roi_left_fraction").value),
            right=float(self.get_parameter("roi_right_fraction").value),
            top=float(self.get_parameter("roi_top_fraction").value),
            bottom=float(self.get_parameter("roi_bottom_fraction").value),
        )
        self._bridge = CvBridge()
        traffic_model_path = str(self.get_parameter("traffic_model_path").value).strip()
        traffic_max_y = float(self.get_parameter("traffic_roi_max_y_fraction").value)
        if traffic_model_path:
            self._traffic_detector = TrafficHousingDetector(
                model_path=traffic_model_path,
                roi=self._detection_roi,
                max_y_fraction=traffic_max_y,
                confidence_threshold=float(
                    self.get_parameter("traffic_model_confidence").value
                ),
                input_size=int(self.get_parameter("traffic_model_input_size").value),
            )
            self._traffic_detector_name = "YOLO housing + R/Y/L/G brightness"
        else:
            self._traffic_detector = TrafficLightDetector(
                roi=self._detection_roi,
                max_y_fraction=traffic_max_y,
            )
            self._traffic_detector_name = "legacy OpenCV"
        self._traffic_filter = TemporalStateFilter(
            window_size=int(self.get_parameter("traffic_filter_window").value),
            required_hits=int(self.get_parameter("traffic_filter_hits").value),
            clear_hits=int(self.get_parameter("traffic_clear_hits").value),
        )
        self._checker_detector = None
        self._checker_filter = None
        if self._checker_enabled:
            self._checker_detector = CheckerboardDetector(roi=self._detection_roi)
            self._checker_filter = TemporalBoolFilter(
                on_hits=int(self.get_parameter("checker_on_hits").value),
                off_hits=int(self.get_parameter("checker_off_hits").value),
            )
        self._frame_index = 0
        self._checker_state = False
        self._checker_detection = None
        self._debug_window_initialized = False

        self._traffic_publisher = self.create_publisher(UInt8, traffic_topic, 10)
        self._checker_publisher = None
        if self._checker_enabled:
            self._checker_publisher = self.create_publisher(Bool, checker_topic, 10)
        # DNN inference is intentionally latest-frame only. A depth-five image
        # queue can otherwise make a valid red decision arrive several old
        # frames late on the vehicle CPU.
        image_qos = QoSProfile(
            history=qos_profile_sensor_data.history,
            depth=1,
            reliability=qos_profile_sensor_data.reliability,
            durability=qos_profile_sensor_data.durability,
        )
        self._subscription = self.create_subscription(
            Image, image_topic, self._image_callback, image_qos
        )
        checker_status = f"enabled -> {checker_topic}" if self._checker_enabled else "disabled"
        self.get_logger().info(
            f"subscribing {image_topic}; publishing {traffic_topic}; "
            f"traffic detector {self._traffic_detector_name}; "
            f"checkerboard {checker_status}"
        )

    def _image_callback(self, message: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as error:  # cv_bridge raises different types by ROS distro.
            self.get_logger().error(f"image conversion failed: {error}")
            return

        traffic = self._traffic_detector.detect(frame)
        stable_state = self._traffic_filter.update(traffic.state)

        if (
            self._checker_enabled
            and self._frame_index % self._checker_stride == 0
            and self._checker_detector is not None
            and self._checker_filter is not None
        ):
            checker = self._checker_detector.detect(frame)
            self._checker_detection = checker
            self._checker_state = self._checker_filter.update(checker.detected)
        self._frame_index += 1

        traffic_message = UInt8()
        traffic_message.data = int(stable_state)
        self._traffic_publisher.publish(traffic_message)

        if self._checker_publisher is not None:
            checker_message = Bool()
            checker_message.data = bool(self._checker_state)
            self._checker_publisher.publish(checker_message)

        if self._show_debug:
            debug = frame.copy()
            x0, y0, x1, y1 = self._traffic_detector.active_bounds(debug.shape)
            shaded = debug.copy()
            shaded[:y0, :] = 0
            shaded[y1:, :] = 0
            shaded[y0:y1, :x0] = 0
            shaded[y0:y1, x1:] = 0
            debug = cv2.addWeighted(debug, 0.45, shaded, 0.55, 0.0)
            cv2.rectangle(debug, (x0, y0), (x1 - 1, y1 - 1), (255, 180, 0), 2)
            state_colors = {
                0: (170, 170, 170),
                1: (40, 40, 255),
                2: (0, 210, 255),
                3: (40, 220, 40),
                4: (255, 210, 30),
            }
            traffic_color = state_colors.get(traffic.state, (255, 255, 255))
            if traffic.bbox is not None:
                x, y, width, height = traffic.bbox
                cv2.rectangle(
                    debug, (x, y), (x + width, y + height), traffic_color, 2
                )
                for label, center in zip(("R", "Y", "L", "G"), traffic.slot_centers):
                    point = (int(round(center[0])), int(round(center[1])))
                    cv2.circle(
                        debug,
                        point,
                        max(3, int(round(traffic.radius))),
                        traffic_color,
                        2,
                    )
                    cv2.putText(
                        debug,
                        label,
                        (point[0] - 5, point[1] - int(round(traffic.radius)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        traffic_color,
                        2,
                        cv2.LINE_AA,
                    )

            if (
                self._checker_enabled
                and self._checker_detection is not None
                and self._checker_detection.bbox is not None
            ):
                x, y, width, height = self._checker_detection.bbox
                cv2.rectangle(debug, (x, y), (x + width, y + height), (255, 0, 255), 3)
                cv2.putText(
                    debug,
                    "CHECKER",
                    (x, max(46, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.rectangle(debug, (0, 0), (debug.shape[1], 42), (20, 20, 20), -1)
            cv2.putText(
                debug,
                (
                    f"traffic raw={traffic.state}  publish={stable_state}   "
                    f"checker={'on' if self._checker_enabled else 'disabled'}  "
                    f"traffic ROI y={y0}..{y1 - 1}"
                ),
                (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                state_colors.get(stable_state, (255, 255, 255)),
                2,
                cv2.LINE_AA,
            )
            try:
                if not self._debug_window_initialized:
                    cv2.namedWindow("traffic_vision", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("traffic_vision", 960, 720)
                    self._debug_window_initialized = True
                cv2.imshow("traffic_vision", debug)
                cv2.waitKey(1)
            except cv2.error as error:
                self.get_logger().error(
                    f"debug window unavailable; disabling it: {error}"
                )
                self._show_debug = False

    def destroy_node(self) -> bool:
        if self._debug_window_initialized:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TrafficVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
