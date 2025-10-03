cmake_minimum_required(VERSION 3.16)
project(homework)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(fmt_test fmt_test.cpp)
add_executable(main main.cpp tools.cpp)

target_link_libraries(fmt_test PRIVATE fmt::fmt)
target_link_libraries(main PRIVATE 
  ${OpenCV_LIBS}
  fmt::fmt
)