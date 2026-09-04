import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    start_camera = DeclareLaunchArgument('start_camera', default_value='true')
    start_lidar = DeclareLaunchArgument('start_lidar', default_value='true')
    start_traffic_vision = DeclareLaunchArgument(
        'start_traffic_vision', default_value='true')
    camera_debug = DeclareLaunchArgument('camera_debug', default_value='false')
    cone_debug = DeclareLaunchArgument('cone_debug', default_value='false')
    traffic_debug = DeclareLaunchArgument('traffic_debug', default_value='false')
    traffic_stop_confirm_frames = DeclareLaunchArgument(
        'traffic_stop_confirm_frames', default_value='2')
    traffic_confirm_frames = DeclareLaunchArgument(
        'traffic_confirm_frames', default_value='3')
    traffic_timeout_sec = DeclareLaunchArgument(
        'traffic_timeout_sec', default_value='0.75')
    require_traffic_for_motion = DeclareLaunchArgument(
        'require_traffic_for_motion', default_value='true')
    require_green_to_start = DeclareLaunchArgument(
        'require_green_to_start', default_value='true')
    require_keyboard_control = DeclareLaunchArgument(
        'require_keyboard_control', default_value='true')
    camera_straight_speed = DeclareLaunchArgument(
        'camera_straight_speed', default_value='23.0')
    camera_curve_speed = DeclareLaunchArgument(
        'camera_curve_speed', default_value='10.0')
    overtake_enabled = DeclareLaunchArgument(
        'overtake_enabled', default_value='true')
    overtake_min_points = DeclareLaunchArgument(
        'overtake_min_points', default_value='5')
    overtake_confirm_scans = DeclareLaunchArgument(
        'overtake_confirm_scans', default_value='2')
    overtake_wall_analysis_forward_max = DeclareLaunchArgument(
        'overtake_wall_analysis_forward_max', default_value='1.30')
    overtake_wall_analysis_lateral_limit = DeclareLaunchArgument(
        'overtake_wall_analysis_lateral_limit', default_value='0.90')
    overtake_vehicle_max_span = DeclareLaunchArgument(
        'overtake_vehicle_max_span', default_value='0.48')
    overtake_wall_reject_span = DeclareLaunchArgument(
        'overtake_wall_reject_span', default_value='0.50')
    overtake_brake_speed = DeclareLaunchArgument(
        'overtake_brake_speed', default_value='-10.0')
    overtake_drive_speed = DeclareLaunchArgument(
        'overtake_drive_speed', default_value='10.0')
    overtake_min_entry_speed = DeclareLaunchArgument(
        'overtake_min_entry_speed', default_value='17.0')
    overtake_straight_max_center_bias = DeclareLaunchArgument(
        'overtake_straight_max_center_bias', default_value='25')
    cone_enter_hold_sec = DeclareLaunchArgument(
        'cone_enter_hold_sec', default_value='0.00')
    cone_bilateral_hold_sec = DeclareLaunchArgument(
        'cone_bilateral_hold_sec', default_value='0.80')
    cone_exit_hold_sec = DeclareLaunchArgument(
        'cone_exit_hold_sec', default_value='0.00')
    cone_reentry_cooldown_sec = DeclareLaunchArgument(
        'cone_reentry_cooldown_sec', default_value='1.0')
    allow_multiple_cone_sections = DeclareLaunchArgument(
        'allow_multiple_cone_sections', default_value='true')
    cone_approach_speed = DeclareLaunchArgument(
        'cone_approach_speed', default_value='8.0')

    usb_cam_params = os.path.join(
        get_package_share_directory('usb_cam'), 'config', 'params.yaml')
    camera_driver = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='xycar_cam',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_camera')),
        respawn=True,
        respawn_delay=2.0,
        arguments=[
            '--ros-args',
            '--log-level', 'error',
            '--log-level', 'compressed_depth_image_transport:=fatal',
            '--log-level', 'image_transport:=fatal',
            '--log-level', 'xycar_cam.image_transport:=fatal',
        ],
        parameters=[
            usb_cam_params,
            {
                'disable_pub_plugins': [
                    'image_transport/compressedDepth',
                    'compressedDepth',
                    'compressed_depth',
                ],
                'image_transport.disable_pub_plugins': [
                    'image_transport/compressedDepth',
                    'compressedDepth',
                ],
                'image_transport_plugins.pub.compressedDepth.enable': False,
                'image_transport_plugins.compressedDepth.enable': False,
            },
        ],
    )

    lidar_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xycar_lidar'),
                'launch',
                'xycar_lidar.launch.py',
            )
        ),
        condition=IfCondition(LaunchConfiguration('start_lidar')),
    )

    final_drive_share = get_package_share_directory('final_drive')
    traffic_vision = Node(
        package='final_drive',
        executable='traffic_vision_node',
        name='traffic_vision',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_traffic_vision')),
        parameters=[
            os.path.join(final_drive_share, 'config', 'traffic_vision.yaml'),
            {
                'image_topic': '/image_raw',
                # Final confirmation is owned by drive_supervisor.  Publish
                # each observed frame here to avoid applying two 3-frame
                # filters in series.
                'traffic_filter_window': 1,
                'traffic_filter_hits': 1,
                'traffic_clear_hits': 1,
                'traffic_roi_max_y_fraction': 0.50,
                'traffic_model_path': os.path.join(
                    final_drive_share,
                    'models',
                    'traffic_housing_v2_best.onnx',
                ),
                'traffic_model_confidence': 0.35,
                'traffic_model_input_size': 640,
                # Keep the archived checkerboard detector out of the live
                # camera callback so it cannot delay traffic-light frames.
                'enable_checkerboard': False,
                'show_debug_window': ParameterValue(
                    LaunchConfiguration('traffic_debug'), value_type=bool),
            },
        ],
    )

    camera_drive = Node(
        package='final_drive',
        executable='camera_drive',
        name='final_camera_drive',
        output='screen',
        parameters=[{
            'debug_view': ParameterValue(
                LaunchConfiguration('camera_debug'), value_type=bool),
            'lane_loss_stop_sec': 1.0,
            'straight_speed': ParameterValue(
                LaunchConfiguration('camera_straight_speed'), value_type=float),
             'minimum_curve_speed': ParameterValue(
                 LaunchConfiguration('camera_curve_speed'), value_type=float),
             'overtake_enabled': ParameterValue(
                 LaunchConfiguration('overtake_enabled'), value_type=bool),
             'lidar_yaw_deg': 180.0,
             'invert_scan_angle': False,
             'overtake_forward_min_m': 0.01,
             'overtake_forward_max_m': 1.5,
             'overtake_lateral_limit_m': 0.40,
             # Classify walls before cropping to the narrow vehicle ROI. This
             # keeps a long S-curve wall from appearing as a compact fragment.
             'overtake_wall_analysis_forward_max_m': ParameterValue(
                 LaunchConfiguration('overtake_wall_analysis_forward_max'),
                 value_type=float),
             'overtake_wall_analysis_lateral_limit_m': ParameterValue(
                 LaunchConfiguration('overtake_wall_analysis_lateral_limit'),
                 value_type=float),
             'overtake_min_points': ParameterValue(
                 LaunchConfiguration('overtake_min_points'), value_type=int),
             'overtake_confirm_scans': ParameterValue(
                 LaunchConfiguration('overtake_confirm_scans'), value_type=int),
             # The target vehicle is about 37 x 25 cm. Keep a diagonal/noise
             # margin, then reject longer connected returns as walls.
             'overtake_vehicle_max_span_m': ParameterValue(
                 LaunchConfiguration('overtake_vehicle_max_span'),
                 value_type=float),
             'overtake_wall_seed_distance_m': 0.16,
             'overtake_wall_expand_distance_m': 0.22,
             'overtake_wall_reject_span_m': ParameterValue(
                 LaunchConfiguration('overtake_wall_reject_span'),
                 value_type=float),
             'overtake_brake_duration_sec': 0.50,
             'overtake_brake_speed': ParameterValue(
                 LaunchConfiguration('overtake_brake_speed'), value_type=float),
             'overtake_drive_speed': ParameterValue(
                 LaunchConfiguration('overtake_drive_speed'), value_type=float),
             'overtake_lane_offset_px': 100,
             'overtake_exit_delay_sec': 1.5,
             'overtake_min_entry_speed': ParameterValue(
                 LaunchConfiguration('overtake_min_entry_speed'),
                 value_type=float),
             # Entry uses only two consecutive real two-lane images. The
             # following thresholds remain diagnostic values and do not gate
             # overtaking.
             'overtake_straight_confirm_frames': 2,
             'overtake_straight_max_center_bias_px': ParameterValue(
                 LaunchConfiguration('overtake_straight_max_center_bias'),
                 value_type=int),
             'overtake_straight_max_angle_deg': 8.0,
             'overtake_straight_min_lane_width_px': 300,
             'overtake_straight_max_lane_width_px': 460,
         }],
    )

    # The proven cone node runs continuously, but its motor output is isolated.
    # Only drive_supervisor is allowed to publish the real /xycar_motor topic.
    cone_drive = Node(
        package='final_drive',
        executable='cone_drive',
        name='final_cone_drive',
        output='screen',
        remappings=[
            ('/xycar_motor', '/final_drive/cone_cmd'),
        ],
        parameters=[{
            'debug_mode': ParameterValue(
                LaunchConfiguration('cone_debug'), value_type=bool),
            'lidar_yaw_deg': 180.0,
            'invert_scan_angle': False,
            'scan_half_angle_deg': 90.0,
	    'drive_speed': 11.0,
	    'one_side_speed': 7.0,
	    'stable_one_side_speed': 8.0,
	    'curve_speed': 8.5,
	    'reversal_speed': 6.5,
            'clearance_speed': 6.5,
            # The supervisor supplies final output smoothing.  A faster
            # internal ramp makes the cone command ready without a 0.2 s
            # mode-switch delay.
            'speed_accel_step': 1.20,
            'vehicle_length': 0.60,
            'vehicle_width': 0.30,
            'wheelbase': 0.30,
            'rear_overhang': 0.15,
            'front_overhang': 0.15,
            'lidar_ahead_of_front_axle': 0.10,
            'body_clearance_margin': 0.08,
        }],
    )

    supervisor = Node(
        package='final_drive',
        executable='drive_supervisor',
        name='final_drive_supervisor',
        output='screen',
        parameters=[{
            'command_timeout_sec': 0.30,
            'lidar_timeout_sec': 0.30,
            'approach_hold_sec': ParameterValue(
                LaunchConfiguration('cone_bilateral_hold_sec'), value_type=float),
            'require_lidar_for_motion': True,
            'cone_active_min_speed': 0.50,
            'cone_enter_hold_sec': ParameterValue(
                LaunchConfiguration('cone_enter_hold_sec'), value_type=float),
            'cone_exit_hold_sec': ParameterValue(
                LaunchConfiguration('cone_exit_hold_sec'), value_type=float),
            'cone_reentry_cooldown_sec': ParameterValue(
                LaunchConfiguration('cone_reentry_cooldown_sec'), value_type=float),
            'allow_multiple_cone_sections': ParameterValue(
                LaunchConfiguration('allow_multiple_cone_sections'),
                value_type=bool),
            'cone_approach_speed': ParameterValue(
                LaunchConfiguration('cone_approach_speed'), value_type=float),
            'steering_step_deg': 12.0,
             'speed_accel_step': 0.60,
             'speed_decel_step': 1.20,
             'camera_reverse_limit': ParameterValue(
                 LaunchConfiguration('overtake_brake_speed'), value_type=float),
            'lidar_yaw_deg': 180.0,
            'invert_scan_angle': False,
            'scan_half_angle_deg': 90.0,
            'traffic_topic': '/traffic_light_state',
            'traffic_stop_confirm_frames': ParameterValue(
                LaunchConfiguration('traffic_stop_confirm_frames'),
                value_type=int),
            'traffic_confirm_frames': ParameterValue(
                LaunchConfiguration('traffic_confirm_frames'), value_type=int),
            'traffic_timeout_sec': ParameterValue(
                LaunchConfiguration('traffic_timeout_sec'), value_type=float),
             'require_traffic_for_motion': ParameterValue(
                 LaunchConfiguration('require_traffic_for_motion'),
                 value_type=bool),
             'require_green_to_start': ParameterValue(
                 LaunchConfiguration('require_green_to_start'),
                 value_type=bool),
            'require_manual_control_for_motion': ParameterValue(
                LaunchConfiguration('require_keyboard_control'),
                value_type=bool),
        }],
    )

    return LaunchDescription([
        start_camera,
        start_lidar,
        start_traffic_vision,
        camera_debug,
        cone_debug,
        traffic_debug,
        traffic_stop_confirm_frames,
        traffic_confirm_frames,
        traffic_timeout_sec,
         require_traffic_for_motion,
         require_green_to_start,
         require_keyboard_control,
         camera_straight_speed,
         camera_curve_speed,
         overtake_enabled,
         overtake_min_points,
         overtake_confirm_scans,
         overtake_wall_analysis_forward_max,
         overtake_wall_analysis_lateral_limit,
         overtake_vehicle_max_span,
         overtake_wall_reject_span,
         overtake_brake_speed,
         overtake_drive_speed,
         overtake_min_entry_speed,
         overtake_straight_max_center_bias,
        cone_enter_hold_sec,
        cone_bilateral_hold_sec,
        cone_exit_hold_sec,
        cone_reentry_cooldown_sec,
        allow_multiple_cone_sections,
        cone_approach_speed,
        camera_driver,
        lidar_driver,
        traffic_vision,
        camera_drive,
        cone_drive,
        supervisor,
    ])
