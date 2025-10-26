cmake_minimum_required(VERSION 3.16.3)

project(final_project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")
find_package(OpenVINO REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})


add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_aim)

####################src###################
add_executable(task_1 src/task_1.cpp)
add_executable(task_2 src/task_2.cpp)
add_executable(task_3 src/task_3.cpp)

target_link_libraries(task_1 ${OpenCV_LIBS} auto_aim fmt::fmt yaml-cpp tools io)
target_link_libraries(task_2 ${OpenCV_LIBS} auto_aim fmt::fmt yaml-cpp tools io)
target_link_libraries(task_3 ${OpenCV_LIBS} auto_aim fmt::fmt yaml-cpp tools io)
