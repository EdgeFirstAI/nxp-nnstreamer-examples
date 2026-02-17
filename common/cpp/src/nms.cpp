/**
 * Copyright 2025 NXP
 * SPDX-License-Identifier: BSD-3-Clause 
 */ 

#include "nms.hpp"
#include <algorithm>


void nms(std::vector<DetectedObject> &results)
{
  size_t boxesSize = results.size();
  if (boxesSize == 0)
    return;

  size_t i;
  #pragma omp parallel for schedule(static) num_threads(4)
  for (i = 0; i < boxesSize; ++i) {
    DetectedObject &a = results.at(i);
    if (a.valid == true) {
      for (size_t j = i + 1; j < boxesSize; ++j) {
        DetectedObject &b = results.at(j);
        if ((b.valid == true) && (iou (a, b) > IOU_THRESHOLD)) {
            b.valid = false;
        }
      }
    }
  }

  /* filter out invalid boxes */
  i = 0;
  do {
    DetectedObject &a = results.at(i);
    if (a.valid == false)
      results.erase(results.begin() + i);
    else
      i++;
  } while (i < results.size());
}


float iou(const DetectedObject &detectedObject1, const DetectedObject &detectedObject2)
{
  int x1 = std::max(detectedObject1.x, detectedObject2.x);
  int y1 = std::max(detectedObject1.y, detectedObject2.y);
  int x2 = std::min(detectedObject1.x + detectedObject1.width, detectedObject2.x + detectedObject2.width);
  int y2 = std::min(detectedObject1.y + detectedObject1.height, detectedObject2.y + detectedObject2.height);
  int w = std::max(0, (x2 - x1 + 1));
  int h = std::max(0, (y2 - y1 + 1));
  float inter = static_cast<float>(w * h);
  float areaA = static_cast<float>(detectedObject1.width * detectedObject1.height);
  float areaB = static_cast<float>(detectedObject2.width * detectedObject2.height);
  float o = inter / (areaA + areaB - inter);
  return std::max(o, 0.0f);
}