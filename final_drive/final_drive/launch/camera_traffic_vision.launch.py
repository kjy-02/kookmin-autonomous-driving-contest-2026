from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = Path(get_package_share_directory("final_drive"))
    camera_share = Path(get_package_share_directory("xycar_cam"))
    image_topic = LaunchConfiguration("image_topic")
    show_debug_window = LaunchConfiguration("show_debug_window")
    return LaunchDescription(
        [
            DeclareLaunchArgument("image_topic", default_value="/image_raw"),
            DeclareLaunchArgument("show_debug_window", default_value="true"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(camera_share / "launch" / "xycar_cam.launch.py")
                )
            ),
            Node(
                package="final_drive",
                executable="traffic_vision_node",
                name="traffic_vision",
                output="screen",
                parameters=[
                    str(package_share / "config" / "traffic_vision.yaml"),
                    {
                        "image_topic": image_topic,
                        "traffic_model_path": str(
                            package_share / "models" / "traffic_housing_v2_best.onnx"
                        ),
                        "traffic_model_confidence": 0.35,
                        "traffic_model_input_size": 640,
                        "show_debug_window": ParameterValue(
                            show_debug_window, value_type=bool
                        ),
                    },
                ],
            ),
        ]
    )
