cmake_minimum_required(VERSION 3.8)
project(final_drive)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

ament_python_install_package(final_drive_vision)

add_executable(traffic_vision_node src/traffic_vision_launcher.cpp)

add_executable(keyboard_stop src/keyboard_stop_node.cpp)
ament_target_dependencies(keyboard_stop
  rclcpp
  std_msgs
)

add_executable(camera_drive src/camera_drive_node.cpp)
target_include_directories(camera_drive PUBLIC ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(camera_drive
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
)
target_link_libraries(camera_drive ${OpenCV_LIBRARIES})

add_executable(drive_supervisor src/drive_supervisor_node.cpp)
ament_target_dependencies(drive_supervisor
  rclcpp
  sensor_msgs
  std_msgs
)

add_executable(cone_drive src/cone_drive_node.cpp)
target_include_directories(cone_drive PUBLIC ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(cone_drive
  rclcpp
  sensor_msgs
  std_msgs
)
target_link_libraries(cone_drive ${OpenCV_LIBRARIES})

install(TARGETS
  traffic_vision_node
  keyboard_stop
  camera_drive
  drive_supervisor
  cone_drive
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/traffic_bag_viewer
  scripts/traffic_bag_evaluate
  scripts/traffic_dataset_prepare
  scripts/traffic_housing_dataset_prepare
  scripts/traffic_housing_color_visualize
  scripts/lidar_cone_inspector.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  models
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(traffic_detector_tests
    test/test_traffic_detectors.py
  )
  ament_add_pytest_test(traffic_dataset_tests
    test/test_prepare_traffic_dataset.py
  )
  ament_add_pytest_test(traffic_housing_dataset_tests
    test/test_prepare_traffic_housing_dataset.py
  )
  ament_add_pytest_test(bag_reader_tests
    test/test_bag_reader.py
  )
endif()

ament_package()
