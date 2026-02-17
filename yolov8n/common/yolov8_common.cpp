/**
 * Copyright 2025 NXP
 * Copyright 2026 EdgeFirst AI (Au-Zone Technologies)
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Shared implementation for YOLOv8n 640x640 demos.
 */

#include "yolov8_common.hpp"
#include <cstdio>


const char *cocoClassNames[NUM_CLASSES] = {
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
  "truck", "boat", "traffic light", "fire hydrant", "stop sign",
  "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
  "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
  "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
  "couch", "potted plant", "bed", "dining table", "toilet", "tv",
  "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
  "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
  "scissors", "teddy bear", "hair drier", "toothbrush"
};


std::vector<std::string> getCocoClassNames() {
  return std::vector<std::string>(cocoClassNames, cocoClassNames + NUM_CLASSES);
}


LetterboxParams calculateLetterbox(int srcWidth, int srcHeight) {
  float scaleX = (float)MODEL_INPUT_SIZE / srcWidth;
  float scaleY = (float)MODEL_INPUT_SIZE / srcHeight;
  float scale = std::min(scaleX, scaleY);
  int scaledW = (int)(srcWidth * scale);
  int scaledH = (int)(srcHeight * scale);
  int padLeft = (MODEL_INPUT_SIZE - scaledW) / 2;
  int padTop = (MODEL_INPUT_SIZE - scaledH) / 2;
  int padRight = MODEL_INPUT_SIZE - scaledW - padLeft;
  int padBottom = MODEL_INPUT_SIZE - scaledH - padTop;
  return {scale, padLeft, padTop, scaledW, scaledH, padRight, padBottom};
}


void printMetric(const char *label, const char *desc, const TimingMetric &metric) {
  if (metric.count > 0) {
    printf("  %s\n", label);
    if (desc)
      printf("     (%s)\n", desc);
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d frames]\n",
           metric.avg(), metric.minMs, metric.maxMs, metric.count);
  }
}
