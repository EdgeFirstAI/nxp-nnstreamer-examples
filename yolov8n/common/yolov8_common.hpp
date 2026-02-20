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

#include <gst/gst.h>
#include <nnstreamer_tensor_quant_meta.h>
#include <glib-unix.h>

#include <cstdint>
#include <map>
#include <string>
#include <sys/time.h>
#include <vector>
#include <algorithm>

/* ─── Model constants ─────────────────────────────────────────────── */

#define MODEL_INPUT_SIZE    640       // Input tensor dimension (square)
#define MODEL_OUTPUT_WIDTH  84        // Output channels: 4 coords + 80 classes
#define NUM_TOTAL_BOXES     8400      // Output boxes: 40*40 + 20*20 + 10*10
#define NUM_COORDINATES     4         // Bounding box attributes (cx, cy, w, h)
#define NUM_CLASSES         80        // COCO class count

/* ─── Post-processing thresholds ──────────────────────────────────── */

#define CONF_THRESHOLD      0.25f
#define NMS_IOU_THRESHOLD   0.45f

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


/* ─── Quantization parameters from GstNnsTensorQuantMeta ─────────── */

/** @brief Per-tensor quantization parameters extracted from GstNnsTensorQuantMeta */
struct QuantParams {
  double scale;
  int64_t zeroPoint;
};

/**
 * @brief Extract quantization parameters for a specific output tensor.
 * @param buffer     GstBuffer with quant meta attached by tensor_filter
 * @param tensorIdx  Output tensor index (0-based)
 * @param out        Output: scale and zeroPoint
 * @return true if quant meta was found and extracted
 */
bool extractQuantParams(GstBuffer *buffer, unsigned int tensorIdx, QuantParams &out);


/** @brief COCO class names (80 classes) as C string array */
extern const char *cocoClassNames[NUM_CLASSES];

/** @brief Get COCO class names as a vector of strings (for NXP-style code) */
std::vector<std::string> getCocoClassNames();

/** @brief Print a timing metric line to stdout */
void printMetric(const char *label, const char *desc, const TimingMetric &metric);


/* ─── Argument Parsing ────────────────────────────────────────────── */

/** @brief Bitmask flags for supported CLI arguments per binary */
enum ArgFlag {
  ARG_MODEL        = (1u << 0),
  ARG_CAMERA       = (1u << 1),
  ARG_VIDEO        = (1u << 2),
  ARG_IMAGE        = (1u << 3),
  ARG_HEADLESS     = (1u << 4),
  ARG_INSTRUMENTED = (1u << 5),
  ARG_NUM_FRAMES   = (1u << 6),
  ARG_PLATFORM     = (1u << 7),
  ARG_SPEED        = (1u << 8),
};

/** @brief Parsed CLI arguments */
struct ParsedArgs {
  std::string model;
  std::string camera;
  std::string video;
  std::string image;
  std::string platformStr;
  bool headless = false;
  bool instrumented = false;
  int numFrames = 0;
  double speed = 1.0;
};

/**
 * @brief Unified CLI argument parser with bitmask-based flag selection.
 *
 * Dynamically builds getopt_long options and help text from supportedFlags.
 * -c is always --camera, -n is --num-frames.
 *
 * @param argc        Argument count from main()
 * @param argv        Argument vector from main()
 * @param supportedFlags  Bitmask of ArgFlag values this binary supports
 * @param binaryDesc  Short description for help text header
 * @param args        Output: parsed arguments
 * @return 0=success, 1=help printed (exit 0), -1=error (exit 1)
 */
int parseArgs(int argc, char **argv, uint32_t supportedFlags,
              const char *binaryDesc, ParsedArgs &args);


/* ─── Bus Callback + SIGINT Handler ───────────────────────────────── */

/** @brief Callback type for printing timing statistics */
typedef void (*TimingPrintFn)(void *appData);

/** @brief Context for the common bus callback */
struct BusCallbackCtx {
  GstElement *pipeline;
  GMainLoop *loop;
  gboolean *playing;
  bool *startedOnce;       // NULL to disable state-change suppression
  bool videoLoop;
  double videoRate;        // 1.0 = normal; uses gst_element_seek for != 1.0
  TimingPrintFn printTiming;
  void *appData;
};

/** @brief Common GStreamer bus message handler */
void commonBusCallback(GstBus *bus, GstMessage *message, gpointer user_data);

/** @brief Common SIGINT handler — prints timing and quits main loop */
gboolean commonSigintHandler(gpointer user_data);


/* ─── Throughput Tracker ──────────────────────────────────────────── */

/** @brief Frame-to-frame interval tracker for throughput measurement */
struct ThroughputTracker {
  TimingMetric metric;
  struct timeval lastFrameTime;
  bool firstFrame;

  void reset();
  void tick(const struct timeval &now);
};


/* ─── PTS-Correlated E2E Tracker ─────────────────────────────────── */

/** @brief PTS-correlated end-to-end latency tracker */
struct PtsTracker {
  std::map<GstClockTime, struct timeval> startTimes;
  GMutex mutex;

  void init();
  void destroy();
  void recordStart(GstBuffer *buffer, const struct timeval &now);
  bool consumeStart(GstBuffer *buffer, struct timeval &startTime);
};


/* ─── Inference Latency Query ─────────────────────────────────────── */

/** @brief Query the latency property from a tensor_filter element */
void queryInferenceLatency(GstElement *tensorFilter, TimingMetric &metric);


/* ─── Source Element Construction ─────────────────────────────────── */

/** @brief Input source type */
enum InputSource { INPUT_CAMERA_V4L2, INPUT_CAMERA_LIBCAMERA, INPUT_VIDEO, INPUT_IMAGE };

/** @brief Determine input source from parsed args */
InputSource determineInputSource(const ParsedArgs &args, bool usesLibcamerasrc);

/**
 * @brief Build a GStreamer source element string.
 * @return g_strdup'd string — caller must g_free()
 */
char *buildSourceElement(InputSource source, const ParsedArgs &args,
                         int srcWidth = SOURCE_WIDTH, int srcHeight = SOURCE_HEIGHT,
                         int numBuffers = 0);


/* ─── i.MX 95 Environment Setup ──────────────────────────────────── */

/** @brief Set LIBCAMERA_PIPELINES_MATCH_LIST and NEUTRON_ENABLE_ZERO_COPY */
void setupImx95Environment();


#endif // YOLOV8_COMMON_HPP
