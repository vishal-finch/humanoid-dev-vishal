"""
full_body_mujoco.launch.py
==========================
ROS 2 launch file for the Angad full-body humanoid in MuJoCo using
ros2_control (mujoco_ros2_control/MujocoSystem plugin).

Pipeline:
  1. Process full_body.xacro → robot_state_publisher (TF / RViz)
  2. Inject XP_robot mesh path into angad_full_body.xml → temp file
  3. Start mujoco_ros2_control with the patched XML + controllers yaml
  4. Spawner nodes activate all controllers after MuJoCo starts
  5. (Optional) Start RViz2

Setup:
  source ~/mujoco_ws/install/setup.bash
  source ~/clean_ws/install/setup.bash
  colcon build --packages-select full_body_mujoco
  ros2 launch full_body_mujoco full_body_mujoco.launch.py

Options:
  rviz:=true/false    (default: true)
  realtime:=1.0       (MuJoCo real-time factor, 0.0 = max speed)
  sim_freq:=500.0     (simulation frequency Hz)
"""

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch.actions import (
    DeclareLaunchArgument,
    LogInfo,
    RegisterEventHandler,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import xacro


def resolve_xp_mesh_dir(pkg_share: str) -> str:
    """Resolve the XP_robot mesh directory without hardcoding a user path."""
    search_roots = [pkg_share, os.path.dirname(__file__)]

    for root in search_roots:
        current = os.path.realpath(root)
        while True:
            mesh_dir = os.path.join(current, 'src', 'XP_robot', 'meshes')
            if os.path.isdir(mesh_dir):
                return mesh_dir

            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

    raise FileNotFoundError(
        'Could not locate XP_robot meshes. Expected to find '
        'src/XP_robot/meshes relative to the workspace root.'
    )


def create_nodes(context: LaunchContext):
    # ── Paths ──────────────────────────────────────────────────────────────────
    pkg_share   = get_package_share_directory('full_body_mujoco')
    xp_mesh_dir = resolve_xp_mesh_dir(pkg_share)

    xacro_file       = os.path.join(pkg_share, 'urdf',   'full_body.xacro')
    controllers_yaml = os.path.join(pkg_share, 'config', 'full_body_controllers.yaml')
    mujoco_xml_templ = os.path.join(pkg_share, 'config', 'angad_full_body.xml')
    rviz_config_file = os.path.join(pkg_share, 'config', 'angad.rviz')

    # ── Launch argument values ─────────────────────────────────────────────────
    rviz_enabled = LaunchConfiguration('rviz').perform(context)
    realtime     = LaunchConfiguration('realtime').perform(context)
    sim_freq     = LaunchConfiguration('sim_freq').perform(context)

    # ── Inject mesh path into MuJoCo XML ──────────────────────────────────────
    # angad_full_body.xml contains the literal string MESHDIR_PLACEHOLDER where
    # the meshdir attribute value should go.  We replace it at launch time so
    # the installed file remains path-independent.
    with open(mujoco_xml_templ, 'r') as f:
        xml_content = f.read()

    xml_patched = xml_content.replace('MESHDIR_PLACEHOLDER', xp_mesh_dir)

    # Write to a temp file (deleted when OS cleans up /tmp)
    tmp = tempfile.NamedTemporaryFile(
        mode='w',
        prefix='angad_full_body_',
        suffix='.xml',
        delete=False,
    )
    tmp.write(xml_patched)
    tmp.flush()
    tmp.close()
    mujoco_xml = tmp.name

    # ── Process URDF xacro → robot_description (TF + ros2_control) ───────────
    robot_description_xml = xacro.process_file(
        xacro_file, 
        mappings={'xp_mesh': 'file://' + xp_mesh_dir}
    ).toprettyxml(indent='  ')
    robot_description = {'robot_description': robot_description_xml}

    # ── 1. Robot State Publisher ───────────────────────────────────────────────
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            robot_description,
            {'use_sim_time': True},
        ],
    )

    # ── 2. mujoco_ros2_control ─────────────────────────────────────────────────
    mujoco_node = Node(
        package='mujoco_ros2_control',
        executable='mujoco_ros2_control',
        output='screen',
        parameters=[
            robot_description,
            controllers_yaml,
            {'simulation_frequency': float(sim_freq)},
            {'realtime_factor':      float(realtime)},
            {'robot_model_path':     mujoco_xml},
            {'show_gui':             True},
            {'use_sim_time':         True},
            {'initial_keyframe_key': 'standing'},  # load standing pose at t=0
        ],
        remappings=[
            ('/controller_manager/robot_description', '/robot_description'),
        ],
    )

    # ── 3. Controller spawners ─────────────────────────────────────────────────
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        name='joint_state_broadcaster_spawner',
        arguments=['joint_state_broadcaster', '-c', '/controller_manager', '--param-file', controllers_yaml],
        output='screen',
    )

    lower_body_controller = Node(
        package='controller_manager',
        executable='spawner',
        name='lower_body_controller_spawner',
        arguments=['lower_body_controller', '-c', '/controller_manager', '--param-file', controllers_yaml],
        output='screen',
    )

    torso_controller = Node(
        package='controller_manager',
        executable='spawner',
        name='torso_controller_spawner',
        arguments=['torso_controller', '-c', '/controller_manager', '--param-file', controllers_yaml],
        output='screen',
    )

    left_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        name='left_arm_controller_spawner',
        arguments=['left_arm_controller', '-c', '/controller_manager', '--param-file', controllers_yaml],
        output='screen',
    )

    right_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        name='right_arm_controller_spawner',
        arguments=['right_arm_controller', '-c', '/controller_manager', '--param-file', controllers_yaml],
        output='screen',
    )

    # ── 3b. Force/Torque sensor broadcaster spawners ──────────────────────────
    ft_broadcaster_names = [
        'contact_force_r_toe_1_broadcaster',
        'contact_force_r_toe_2_broadcaster',
        'contact_force_r_heel_1_broadcaster',
        'contact_force_r_heel_2_broadcaster',
        'contact_force_l_toe_1_broadcaster',
        'contact_force_l_toe_2_broadcaster',
        'contact_force_l_heel_1_broadcaster',
        'contact_force_l_heel_2_broadcaster',
    ]
    ft_broadcaster_nodes = [
        Node(
            package='controller_manager',
            executable='spawner',
            name=f'{name}_spawner',
            arguments=[name, '-c', '/controller_manager', '--param-file', controllers_yaml],
            output='screen',
        )
        for name in ft_broadcaster_names
    ]

    # ── 4. RViz2 (optional) ────────────────────────────────────────────────────
    rviz2 = Node(
        condition=IfCondition(rviz_enabled),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': True}],
    )

    # ── 4b. Static TF Publisher (odom -> base_link) ────────────────────────────
    # Provides a default fixed frame for RViz and satisfies controller TF lookups
    odom_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
        output='screen'
    )

    # ── 5. Standing pose publisher — commands initial pose to all controllers ────
    hold_standing_pose = Node(
        package='full_body_mujoco',
        executable='foot_force_array_publisher.py',
        name='foot_force_array_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # ── Event chain: spawn controllers after MuJoCo node starts ───────────────
    load_controllers = RegisterEventHandler(
        OnProcessStart(
            target_action=mujoco_node,
            on_start=[
                LogInfo(msg='[full_body_mujoco] MuJoCo started. Spawning controllers...'),
                joint_state_broadcaster,
                lower_body_controller,
                torso_controller,
                left_arm_controller,
                right_arm_controller,
                rviz2,
                odom_static_tf,
                hold_standing_pose,
            ] + ft_broadcaster_nodes,
        )
    )

    return [
        robot_state_publisher,
        mujoco_node,
        load_controllers,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Launch RViz2 for visualisation',
        ),
        DeclareLaunchArgument(
            'realtime',
            default_value='1.0',
            description='MuJoCo real-time factor (0.0 = max speed)',
        ),
        DeclareLaunchArgument(
            'sim_freq',
            default_value='500.0',
            description='Simulation frequency in Hz (must match MuJoCo timestep)',
        ),
        OpaqueFunction(function=create_nodes),
    ])
