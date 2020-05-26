cmake_minimum_required(VERSION 2.8.3)
project(vimantic)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
  genmsg
  actionlib_msgs
  actionlib
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  SemanticObject.msg
  SemanticObjects.msg
  SemanticRoom.msg
  SemanticRoomScore.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
  actionlib_msgs
)

## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
   CATKIN_DEPENDS
   geometry_msgs
   std_msgs
   sensor_msgs
   message_runtime
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


