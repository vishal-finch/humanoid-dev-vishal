cmake_minimum_required(VERSION 3.8)
project(full_body_mujoco)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(ament_cmake REQUIRED)


install(DIRECTORY
  config
  launch
  urdf
  scripts
  DESTINATION share/${PROJECT_NAME}
)


# Make Python scripts executable
install(PROGRAMS
  scripts/foot_force_array_publisher.py
  scripts/zmp_calculation.py
  scripts/com_calculation.py
  scripts/support_polygon.py
  scripts/pd_control.py
  scripts/stable_crouch_positon.py
  scripts/angad_robot_params.py
  DESTINATION lib/${PROJECT_NAME}
)

# Message generation
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FootSensor.msg"
  "msg/FootSensorArray.msg"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
