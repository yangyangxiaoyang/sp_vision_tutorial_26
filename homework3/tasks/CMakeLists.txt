cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")  # 修改为你的OpenVINO安装路径
find_package(OpenVINO REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenVINO_INCLUDE_DIRS})

add_library(auto_buff OBJECT 
    buff_type.cpp
    buff_detector.cpp
    yolo11_buff.cpp
    buff_solver.cpp
)

target_link_libraries(auto_buff openvino::runtime ${CERES_LIBRARIES})