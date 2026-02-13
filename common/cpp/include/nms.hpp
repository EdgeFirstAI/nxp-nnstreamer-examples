/**
 * Copyright 2025 NXP
 * SPDX-License-Identifier: BSD-3-Clause 
 */ 

#ifndef CPP_NMS_H_
#define CPP_NMS_H_

#include <vector>

#define IOU_THRESHOLD 0.7

/** @brief Represents a detected object.
 * The coordinates, class ID and dimensions are used to draw a rectangle around
 * each detected object, considered valid, with its name. 
 * The drawing is done on the input image with cairo library, and the resulting
 * buffer is displayed.
 */
typedef struct {
  // if the detected object should be displayed
  bool valid = false;
  // class ID of the detected object
  int classId;
  // the X coordinate of the top left corner of the rectangle
  int x;
  // the Y coordinate of the top left corner of the rectangle
  int y;
  // the width of the rectangle (pixels)
  int width;
  // the height of the rectangle (pixels)
  int height;
  // probability score between 0 and 1
  float prob;
} DetectedObject;


/**
 * @brief Apply NMS to the given results.
 * @param results: [in/out] the results to be filtered with nms.
 */
void nms(std::vector<DetectedObject> &results);


/**
 * @brief Calculate the Intersection over Union (IoU) of two detected objects.
 * @param detectedObject1: the first detected object, represented as a DetectedObject structure.
 * @param detectedObject2: the second detected object, represented as a DetectedObject structure.
 * 
 * @return The IoU value as a float, ranging from 0.0 to 1.0.
 */
float iou(const DetectedObject &detectedObject1, const DetectedObject &detectedObject2);

#endif