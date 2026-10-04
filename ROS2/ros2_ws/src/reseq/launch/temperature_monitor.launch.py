from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='reseq',
            executable='temperature_sensor',
            name='temperature_sensor'
        ),

        Node(
            package='reseq',
            executable='temperature_logger',
            name='temperature_logger'
        ),
    ])

