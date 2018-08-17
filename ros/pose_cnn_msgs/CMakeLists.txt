cmake_minimum_required(VERSION 2.8)
project(pose_cnn_msgs)

# generate the messages and the actionlib server for ROS
find_package(catkin REQUIRED genmsg sensor_msgs std_msgs)

add_message_files(DIRECTORY msg
                  FILES PoseCNNMsg.msg
)

generate_messages(DEPENDENCIES sensor_msgs std_msgs)


catkin_package(CATKIN_DEPENDS
  message_runtime
  std_msgs
  sensor_msgs
 )