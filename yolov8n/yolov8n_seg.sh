#!/bin/bash
#
# Copyright 2026 EdgeFirst AI (Au-Zone Technologies)
# SPDX-License-Identifier: BSD-3-Clause
#
# YOLOv8n Instance Segmentation Demo — EdgeFirst Overlay Pipeline
#
# Demonstrates the edgefirstoverlay element with a TFLite segmentation model
# on the i.MX 8M Plus using VX-Delegate + CameraAdaptor + DMA-BUF zero-copy.
#
# Usage:
#   yolov8n_seg.sh [model_path]
#
# Arguments:
#   model_path  Path to YOLOv8n-seg TFLite model
#               (default: /opt/edgefirst/models/yolov8n-seg_640x640.tflite)
#

MODEL="${1:-/opt/edgefirst/models/yolov8n-seg_640x640.tflite}"

if [ ! -f "$MODEL" ]; then
  echo "ERROR: Model file not found: $MODEL" >&2
  exit 1
fi

echo "Model: $MODEL"

gst-launch-1.0 -e \
  v4l2src device=/dev/video3 ! video/x-raw,width=1920,height=1080 \
  ! tee name=t \
  t. ! queue leaky=2 max-size-buffers=2 \
     ! edgefirstoverlay name=ov score-threshold=0.25 iou-threshold=0.45 \
     ! waylandsink \
  t. ! queue leaky=2 max-size-buffers=2 \
     ! edgefirstcameraadaptor model-width=640 model-height=640 model-dtype=uint8 letterbox=true \
     ! tensor_filter framework=tensorflow2-lite model="$MODEL" \
         custom=Delegate:External,ExtDelegateLib:libvx_delegate.so,CameraAdaptor:rgba,DmaBuf:true \
         latency=1 \
     ! ov.tensors
