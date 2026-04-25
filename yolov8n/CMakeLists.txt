# Copyright 2025 NXP
# Copyright 2026 EdgeFirst AI (Au-Zone Technologies)
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.10)
project(YOLOv8n_EdgeFirst)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Required dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GST_ALLOC REQUIRED gstreamer-allocators-1.0)
pkg_check_modules(CAIRO REQUIRED cairo)
pkg_check_modules(NNSTREAMER REQUIRED nnstreamer)
pkg_check_modules(EDGEFIRST_GST REQUIRED edgefirst-gstreamer-1.0)
find_package(OpenMP REQUIRED)

# Include directories
include_directories(
  ${GSTREAMER_INCLUDE_DIRS}
  ${GST_ALLOC_INCLUDE_DIRS}
  ${CAIRO_INCLUDE_DIRS}
  ${NNSTREAMER_INCLUDE_DIRS}
  ${EDGEFIRST_GST_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/include
)

link_directories(
  ${GSTREAMER_LIBRARY_DIRS}
  ${GST_ALLOC_LIBRARY_DIRS}
  ${CAIRO_LIBRARY_DIRS}
  ${NNSTREAMER_LIBRARY_DIRS}
  ${EDGEFIRST_GST_LIBRARY_DIRS}
)

# Shared common library
set(COMMON_SRC common/yolov8_common.cpp)

# --- yolov8n_reference ---
# Standard NXP pipeline baseline for both i.MX 95 and i.MX 8M Plus.
# Uses manual NMS (nms.cpp) — no EdgeFirst dependency.
add_executable(yolov8n_reference
  yolov8n_reference.cpp
  ${COMMON_SRC}
  ${CMAKE_CURRENT_SOURCE_DIR}/../common/cpp/src/nms.cpp
)
target_link_libraries(yolov8n_reference
  ${GSTREAMER_LIBRARIES}
  ${CAIRO_LIBRARIES}
  ${NNSTREAMER_LIBRARIES}
  "${OpenMP_CXX_FLAGS}"
)
target_compile_options(yolov8n_reference PRIVATE "${OpenMP_CXX_FLAGS}")

# --- yolov8n ---
# Unified EdgeFirst overlay pipeline for detection and instance segmentation.
# Supports TFLite (VX/Neutron delegates) and Ara-2 backends on all platforms.
# Model type (detection vs segmentation) is auto-detected from tensor shapes.
add_executable(yolov8n
  yolov8n.cpp
  ${COMMON_SRC}
)
target_link_libraries(yolov8n
  ${GSTREAMER_LIBRARIES}
  ${GST_ALLOC_LIBRARIES}
  ${NNSTREAMER_LIBRARIES}
  ${EDGEFIRST_GST_LIBRARIES}
  edgefirst_hal
)

# --- yolov8n_ara2_reference ---
# Ara-2 NPU reference baseline with standard NXP preprocessing.
# Manual dequant + NMS — no EdgeFirst dependency.
add_executable(yolov8n_ara2_reference
  yolov8n_ara2_reference.cpp
  ${COMMON_SRC}
)
target_link_libraries(yolov8n_ara2_reference
  ${GSTREAMER_LIBRARIES}
  ${CAIRO_LIBRARIES}
  ${NNSTREAMER_LIBRARIES}
)

