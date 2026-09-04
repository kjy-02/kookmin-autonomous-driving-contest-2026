import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # 1. LiDAR 드라이버만 구동 (카메라 에러 메시지 완벽 차단)
    lidar_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xycar_lidar'),
                'launch/xycar_lidar.launch.py')),
        condition=IfCondition(LaunchConfiguration('start_lidar')),
    )

    start_lidar_arg = DeclareLaunchArgument(
        'start_lidar',
        default_value='true',
        description='Start the real LiDAR driver (false for rosbag playback)',
    )

    # 2. 디버그 창 실행 파라미터 (기본값 true)
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='true',
        description='Enable debug view'
    )
    drive_speed_arg = DeclareLaunchArgument('drive_speed', default_value='9.5')
    one_side_speed_arg = DeclareLaunchArgument('one_side_speed', default_value='6.0')
    stable_one_side_speed_arg = DeclareLaunchArgument(
        'stable_one_side_speed', default_value='7.0')
    stable_one_side_scans_arg = DeclareLaunchArgument(
        'stable_one_side_scans', default_value='3')
    curve_speed_arg = DeclareLaunchArgument('curve_speed', default_value='7.5')
    reversal_speed_arg = DeclareLaunchArgument('reversal_speed', default_value='5.5')
    clearance_speed_arg = DeclareLaunchArgument('clearance_speed', default_value='5.5')
    full_speed_angle_deg_arg = DeclareLaunchArgument('full_speed_angle_deg', default_value='8.0')
    slow_speed_angle_deg_arg = DeclareLaunchArgument('slow_speed_angle_deg', default_value='28.0')
    recovery_speed_arg = DeclareLaunchArgument('recovery_speed', default_value='3.0')
    speed_accel_step_arg = DeclareLaunchArgument('speed_accel_step', default_value='0.35')
    speed_decel_step_arg = DeclareLaunchArgument('speed_decel_step', default_value='0.45')
    singleton_hold_scans_arg = DeclareLaunchArgument('singleton_hold_scans', default_value='5')
    empty_hold_scans_arg = DeclareLaunchArgument('empty_hold_scans', default_value='2')
    rear_axle_lookahead_arg = DeclareLaunchArgument('rear_axle_lookahead', default_value='0.90')
    steering_gain_arg = DeclareLaunchArgument('steering_gain', default_value='2.60')
    target_filter_alpha_arg = DeclareLaunchArgument('target_filter_alpha', default_value='0.55')
    steering_filter_alpha_arg = DeclareLaunchArgument('steering_filter_alpha', default_value='0.55')
    target_reversal_alpha_arg = DeclareLaunchArgument('target_reversal_alpha', default_value='0.85')
    steering_reversal_alpha_arg = DeclareLaunchArgument('steering_reversal_alpha', default_value='0.85')
    steering_reversal_delta_deg_arg = DeclareLaunchArgument(
        'steering_reversal_delta_deg', default_value='10.0')
    steering_reversal_min_angle_deg_arg = DeclareLaunchArgument(
        'steering_reversal_min_angle_deg', default_value='12.0')
    steering_reversal_memory_scans_arg = DeclareLaunchArgument(
        'steering_reversal_memory_scans', default_value='10')
    s_turn_hold_scans_arg = DeclareLaunchArgument('s_turn_hold_scans', default_value='7')

    # 3. 라바콘 주행 노드
    cone_drive_node = Node(
        package='final_drive',
        executable='cone_drive',
        name='kuac_cone_driver',
        parameters=[{
            'debug_mode': ParameterValue(LaunchConfiguration('debug_mode'), value_type=bool),
            'drive_speed': ParameterValue(LaunchConfiguration('drive_speed'), value_type=float),
            'one_side_speed': ParameterValue(LaunchConfiguration('one_side_speed'), value_type=float),
            'stable_one_side_speed': ParameterValue(
                LaunchConfiguration('stable_one_side_speed'), value_type=float),
            'stable_one_side_scans': ParameterValue(
                LaunchConfiguration('stable_one_side_scans'), value_type=int),
            'curve_speed': ParameterValue(LaunchConfiguration('curve_speed'), value_type=float),
            'reversal_speed': ParameterValue(
                LaunchConfiguration('reversal_speed'), value_type=float),
            'clearance_speed': ParameterValue(
                LaunchConfiguration('clearance_speed'), value_type=float),
            'full_speed_angle_deg': ParameterValue(
                LaunchConfiguration('full_speed_angle_deg'), value_type=float),
            'slow_speed_angle_deg': ParameterValue(
                LaunchConfiguration('slow_speed_angle_deg'), value_type=float),
            'recovery_speed': ParameterValue(LaunchConfiguration('recovery_speed'), value_type=float),
            'speed_accel_step': ParameterValue(
                LaunchConfiguration('speed_accel_step'), value_type=float),
            'speed_decel_step': ParameterValue(
                LaunchConfiguration('speed_decel_step'), value_type=float),
            'singleton_hold_scans': ParameterValue(LaunchConfiguration('singleton_hold_scans'), value_type=int),
            'empty_hold_scans': ParameterValue(LaunchConfiguration('empty_hold_scans'), value_type=int),
            'lidar_yaw_deg': 180.0,
            'invert_scan_angle': False,
            'scan_half_angle_deg': 90.0,
            'min_cluster_points': 3,
            'cone_base_radius': 0.07,
            'cone_diameter_tolerance': 0.02,
            'cone_radius_fit_tolerance': 0.02,
            'max_cluster_span': 0.16,
            'vehicle_length': 0.60,
            'vehicle_width': 0.30,
            'wheelbase': 0.30,
            'rear_overhang': 0.15,
            'front_overhang': 0.15,
            'lidar_ahead_of_front_axle': 0.10,
            'body_clearance_margin': 0.08,
            'near_cone_check_forward': 0.30,
            'max_clearance_bias': 0.08,
            'one_side_center_offset': 0.38,
            'rear_axle_lookahead': ParameterValue(
                LaunchConfiguration('rear_axle_lookahead'), value_type=float),
            'steering_gain': ParameterValue(
                LaunchConfiguration('steering_gain'), value_type=float),
            'target_filter_alpha': ParameterValue(
                LaunchConfiguration('target_filter_alpha'), value_type=float),
            'steering_filter_alpha': ParameterValue(
                LaunchConfiguration('steering_filter_alpha'), value_type=float),
            'target_reversal_alpha': ParameterValue(
                LaunchConfiguration('target_reversal_alpha'), value_type=float),
            'steering_reversal_alpha': ParameterValue(
                LaunchConfiguration('steering_reversal_alpha'), value_type=float),
            'steering_reversal_delta_deg': ParameterValue(
                LaunchConfiguration('steering_reversal_delta_deg'), value_type=float),
            'steering_reversal_min_angle_deg': ParameterValue(
                LaunchConfiguration('steering_reversal_min_angle_deg'), value_type=float),
            'steering_reversal_memory_scans': ParameterValue(
                LaunchConfiguration('steering_reversal_memory_scans'), value_type=int),
            's_turn_hold_scans': ParameterValue(
                LaunchConfiguration('s_turn_hold_scans'), value_type=int),
        }],
        output='screen',
    )

    return LaunchDescription([
        start_lidar_arg,
        lidar_include,
        debug_mode_arg,
        drive_speed_arg,
        one_side_speed_arg,
        stable_one_side_speed_arg,
        stable_one_side_scans_arg,
        curve_speed_arg,
        reversal_speed_arg,
        clearance_speed_arg,
        full_speed_angle_deg_arg,
        slow_speed_angle_deg_arg,
        recovery_speed_arg,
        speed_accel_step_arg,
        speed_decel_step_arg,
        singleton_hold_scans_arg,
        empty_hold_scans_arg,
        rear_axle_lookahead_arg,
        steering_gain_arg,
        target_filter_alpha_arg,
        steering_filter_alpha_arg,
        target_reversal_alpha_arg,
        steering_reversal_alpha_arg,
        steering_reversal_delta_deg_arg,
        steering_reversal_min_angle_deg_arg,
        steering_reversal_memory_scans_arg,
        s_turn_hold_scans_arg,
        cone_drive_node,
    ])
