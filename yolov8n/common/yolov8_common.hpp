/**
 * Copyright 2025 NXP
 * Copyright 2026 EdgeFirst AI (Au-Zone Technologies)
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Shared constants, types, and utilities for YOLOv8n 640x640 demos.
 * Used by all yolov8n binaries (reference, imx95, imx8mp, ara2).
 */

#ifndef YOLOV8_COMMON_HPP
#define YOLOV8_COMMON_HPP

#include <sys/time.h>
#include <string>
#include <vector>
#include <algorithm>

/* ─── Model constants ─────────────────────────────────────────────── */

#define MODEL_INPUT_SIZE    640       // Input tensor dimension (square)
#define MODEL_OUTPUT_WIDTH  84        // Output channels: 4 coords + 80 classes
#define NUM_TOTAL_BOXES     8400      // Output boxes: 40*40 + 20*20 + 10*10
#define NUM_COORDINATES     4         // Bounding box attributes (cx, cy, w, h)
#define NUM_CLASSES         80        // COCO class count

/* ─── Quantization parameters (from model inspection via Netron/eIQ) ── */

#define ZERO_POINT          (-128)
#define SCALE_FACTOR        0.00390632f

/* ─── Post-processing thresholds ──────────────────────────────────── */

#define CONF_THRESHOLD      0.25f
#define NMS_IOU_THRESHOLD   0.45f
// Threshold in quantized domain: quantized > (real / scale) + zero_point
#define QUANTIZED_THRESHOLD static_cast<int>((CONF_THRESHOLD / SCALE_FACTOR) + ZERO_POINT)

/* ─── Default source video dimensions ─────────────────────────────── */

#define SOURCE_WIDTH  1920
#define SOURCE_HEIGHT 1080


/** @brief Helper to get time difference in milliseconds */
static inline double timeDiffMs(const struct timeval &start, const struct timeval &end) {
  return 1000.0 * (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000.0;
}


/** @brief Single timing metric with min/max/avg tracking */
typedef struct {
  double totalMs;
  int count;
  double minMs;
  double maxMs;

  void reset() {
    totalMs = 0;
    count = 0;
    minMs = 1e9;
    maxMs = 0;
  }

  void record(double ms) {
    if (ms > 0 && ms < 1000) {  // Filter bogus values
      totalMs += ms;
      count++;
      if (ms < minMs) minMs = ms;
      if (ms > maxMs) maxMs = ms;
    }
  }

  double avg() const { return count > 0 ? totalMs / count : 0; }
} TimingMetric;


/** @brief Letterbox parameters for coordinate mapping */
struct LetterboxParams {
  float scale;    // Scale factor applied to source image
  int padX;       // Left padding in model input space
  int padY;       // Top padding in model input space
  int scaledW;    // Scaled source width
  int scaledH;    // Scaled source height
  int padRight;   // Right padding
  int padBottom;  // Bottom padding
};

/** @brief Calculate letterbox parameters for fitting source into model input.
 *  Maintains aspect ratio with centered padding. */
LetterboxParams calculateLetterbox(int srcWidth, int srcHeight);


/** @brief COCO class names (80 classes) as C string array */
extern const char *cocoClassNames[NUM_CLASSES];

/** @brief Get COCO class names as a vector of strings (for NXP-style code) */
std::vector<std::string> getCocoClassNames();

/** @brief Print a timing metric line to stdout */
void printMetric(const char *label, const char *desc, const TimingMetric &metric);

#endif // YOLOV8_COMMON_HPP
