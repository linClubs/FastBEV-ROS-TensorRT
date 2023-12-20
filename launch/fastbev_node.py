from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    node = Node(
            package='fastbev',
            executable='fastbev_node',
            name='fastbev_node'
        )
    
    return LaunchDescription([
       node
    ])