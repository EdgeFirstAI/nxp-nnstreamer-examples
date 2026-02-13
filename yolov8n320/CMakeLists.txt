# Copyright 2025 NXP
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.10)
project(YOLOv8_demo)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add GStreamer library
find_package(PkgConfig)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-app-1.0)
include_directories(${GSTREAMER_INCLUDE_DIRS})
link_directories(${GSTREAMER_LIBRARY_DIRS})

# Add Cairo library
pkg_check_modules(CAIRO REQUIRED cairo)
include_directories(${CAIRO_INCLUDE_DIRS})
link_directories(${CAIRO_LIBRARY_DIRS})

find_package(OpenMP REQUIRED)

# Add Custom library
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/include/)

# YOLOv8 N 320
add_executable(yolov8n_320_tflite yolov8n_320_tflite.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/src/nms.cpp)
target_link_libraries(yolov8n_320_tflite ${GSTREAMER_LIBRARIES} ${CAIRO_LIBRARIES} tensorflow-lite "${OpenMP_CXX_FLAGS}")
set_target_properties(yolov8n_320_tflite PROPERTIES RUNTIME_OUTPUT_DIRECTORY ./detection)
target_compile_options(yolov8n_320_tflite PRIVATE "${OpenMP_CXX_FLAGS}")
