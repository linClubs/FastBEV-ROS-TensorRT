#!/bin/bash
cmake_minimum_required(VERSION 2.8.12)
project(fastbev)

set(CMAKE_BUILD_PYTHON $ENV{USE_Python})
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -Xcompiler -std=c++14,-Wextra,-Wall,-Wno-deprecated-declarations,-O3 -DENABLE_TEXT_BACKEND_STB")
set(CUDA_NVCC_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
#   -gencode arch=compute_$ENV{CUDASM},code=compute_$ENV{CUDASM}
# )


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

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  
  message_filters
  sensor_msgs
  cv_bridge
  
  tf2
  # geometry_msgs
  # jsk_recognition_msgs
  # vision_msgs
  roslib
)


include_directories(
  ${Protobuf_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS}
  third_party/stb
  third_party/cuOSD/src
  third_party/3DSparseConvolution/libspconv/include
  src/common
  src/

  include_ros
  ${catkin_INCLUDE_DIRS}
)

catkin_package()

link_directories(
  ${TensorRT_LIBS}
  ${CUDA_LIBS}
  third_party/3DSparseConvolution/libspconv/lib/${arch}
  build
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

####################### libfastbev_core.so ##########################
cuda_add_library(fastbev_core SHARED 
  ${CORE_FILES}
)

target_link_libraries(fastbev_core
  libcudart.so            # CUDA_LIBS
  libcublasLt.so          # CUDA_LIBS
  libnvinfer.so           # TensorRT_LIBS
  libnvinfer_plugin.so    # TensorRT_LIBS
  libspconv.so            #  3DSparseConvolution
  ${Protobuf_LIBRARIES}
  ${OpenCV_LIBS}
 
)

####################### fastbev program #############################
cuda_add_executable(fastbev 
  src/main.cpp
  ${VISUALIZE_FILES}
)

target_link_libraries(fastbev
  fastbev_core
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  ${OpenCV_LIBS}
)

cuda_add_executable(01test
  src/01test.cpp
  ${VISUALIZE_FILES}
)

target_link_libraries(01test
  fastbev_core
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  ${OpenCV_LIBS}
)

# -----------stb2cv----------
# add_executable(stb2cv src/stb2cv.cpp)
# target_link_libraries(stb2cv ${OpenCV_LIBS})
# ---------------------------

cuda_add_executable(demo
  src_ros/demo.cpp
  src_ros/fastbev_plugin.cpp
  ${VISUALIZE_FILES}
)

target_link_libraries(demo
  fastbev_core
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  ${OpenCV_LIBS}
)

cuda_add_executable(fastbev_node
  src_ros/fastbev_ros_node.cpp
  src_ros/fastbev_ros.cpp
  src_ros/fastbev_plugin.cpp
  ${VISUALIZE_FILES}
)

target_link_libraries(fastbev_node
  fastbev_core
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
)