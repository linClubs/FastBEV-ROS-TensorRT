#!/bin/bash
cmake_minimum_required(VERSION 3.5)
project(fastbev)

# add_definitions(-w)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3 -DENABLE_TEXT_BACKEND_STB")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)

  endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
set(CUDA_LIBS /usr/local/cuda/lib64)

set(CUDASM 86)
set(TensorRT_ROOT /home/lin/software/TensorRT-8.5.3.1)
# set(TensorRT_ROOT ~/share/software/TensorRT-8.5.3.1)
set(TensorRT_INCLUDE_DIRS ${TensorRT_ROOT}/include)
set(TensorRT_LIBS ${TensorRT_ROOT}/lib/)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_${CUDASM},code=compute_${CUDASM})

find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)
find_package(OpenCV)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

include_directories(
  ${Protobuf_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  
  third_party/stb
  third_party/cuOSD/src
  third_party/3DSparseConvolution/libspconv/include
  src/common
  src/

  include_ros
)

link_directories(
  ${TensorRT_LIBS}
  ${CUDA_LIBS}
  third_party/3DSparseConvolution/libspconv/lib/${arch}
  # build
)

file(GLOB_RECURSE CORE_FILES 
  src/fastbev/*.cu 
  src/fastbev/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES 
  third_party/cuOSD/src/*.cpp
  third_party/cuOSD/src/*.cu
  src/common/visualize.cu
)

# ####################### 生成共享库libfastbev_core.so ##########################
# cuda_add_library(fastbev_core SHARED 
#   ${CORE_FILES}
# )

# target_link_libraries(fastbev_core
#   libcudart.so            # CUDA_LIBS
#   libcublasLt.so          # CUDA_LIBS
#   libnvinfer.so           # TensorRT_LIBS
#   libnvinfer_plugin.so    # TensorRT_LIBS
#   libspconv.so            #  3DSparseConvolution
#   ${Protobuf_LIBRARIES}
#   ${OpenCV_LIBS}
# )


cuda_add_executable(fastbev_node
  src_ros/fastbev_ros_node.cpp
  src_ros/fastbev_ros.cpp
  src_ros/fastbev_plugin.cpp
  
  ${VISUALIZE_FILES}

  ${CORE_FILES}
)

target_link_libraries(fastbev_node
  # fastbev_core
  
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  ${OpenCV_LIBS}

  libcudart.so            # CUDA_LIBS
  libcublasLt.so          # CUDA_LIBS
)

ament_target_dependencies(fastbev_node
  rclcpp 
  std_msgs
  sensor_msgs
  cv_bridge
  message_filters
)

# c程序需要加下面2句 
install(TARGETS
  fastbev_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})