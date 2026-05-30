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

/* ─── Segmentation model constants ───────────────────────────────── */

#define MODEL_SEG_OUTPUT_WIDTH  116   // 4 bbox + 80 classes + 32 mask coefficients
#define MODEL_SEG_NUM_PROTOS    32    // Prototype mask channels
#define MODEL_SEG_PROTO_DIM     160   // Prototype mask spatial dimension
#define MODEL_SEG_NUM_OUTPUTS   4     // Split output tensors (classes, protos, bbox, coeffs)

/* ─── Post-processing thresholds ──────────────────────────────────── */

#define CONF_THRESHOLD      0.25f
#define NMS_IOU_THRESHOLD   0.45f
#define MASK_OPACITY        0.5f    // Segmentation mask blend alpha [0..1]

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
  ARG_SEG          = (1u << 9),
  ARG_COMPUTE      = (1u << 10),
  ARG_DETECTIONS   = (1u << 11),
  ARG_COLOR_MODE   = (1u << 12),
  ARG_SAVE_FRAME   = (1u << 13),
};

/** @brief Parsed CLI arguments */
struct ParsedArgs {
  std::string model;
  std::string camera;
  std::string video;
  std::string image;
  std::string platformStr;
  std::string compute;    // edgefirstcameraadaptor backend: "auto", "opengl", "g2d", "cpu"
  bool headless = false;
  bool instrumented = false;
  bool detections = false;
  std::string colorMode;   // edgefirstoverlay color-mode: class|instance|track
  std::string saveFrame;   // save one frame to PNG at this path
  int saveFrameDelay = 750; // frame number at which to save (default ~30s at 25fps)
  int numFrames = 0;
  double speed = 1.0;
  bool segmentation = false;
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


/* ─── Segmentation Overlay Context ────────────────────────────────── */

struct hal_image_processor;
struct hal_tensor;

/** @brief Context for GPU-accelerated segmentation mask overlay rendering
 *         (used by yolov8n_imx8mp / yolov8n_imx95 CPU-copy binaries). */
struct SegContext {
  hal_image_processor *processor;   // GPU image processor
  hal_tensor *overlay;              // PBO-backed RGBA overlay tensor
  GMutex mutex;                     // Protects overlay between NN and draw threads
  bool overlayReady;                // True when overlay has been rendered
  bool segMode;                     // True when seg model detected at runtime
  int overlayWidth;
  int overlayHeight;
};

/* ─── Inference Latency Query ─────────────────────────────────────── */

/** @brief Query the latency property from a tensor_filter element */
void queryInferenceLatency(GstElement *tensorFilter, TimingMetric &metric);


/* ─── Per-element pipeline probes (full + per-element latency) ────── */

/**
 * @brief Per-element pipeline latency probes for the standard NN branch.
 *
 * Installs pad probes on the standard EdgeFirst optimized NN branch
 * elements (queue → preproc → tensor_filter → tensor_sink) to report
 * per-element sink→src latencies plus a race-safe full pipeline latency
 * (queue sink → end of post-processing).
 *
 * Pipeline shape this targets:
 *
 *   src → queue(name=thread-nn) → edgefirstcameraadaptor(name=preproc)
 *      → tensor_filter(name=tfilter) → tensor_sink(name=inferenceOutput)
 *
 * Race safety: queueSinkStart is captured on the upstream thread when a
 * new buffer arrives at the queue input. Because that can fire while
 * the downstream chain is still processing the previous buffer, we
 * snapshot queueSinkStart into currentFrameQueueStart inside the queue
 * src probe (which fires on the downstream thread the moment the
 * buffer leaves the queue). The tensor_sink callback then reads
 * currentFrameQueueStart, which is stable for the lifetime of the
 * downstream processing of one buffer.
 */
struct PipelineProbes {
  // Per-stage timing metrics (sink → src for each element)
  TimingMetric queueDwell;       // queue sink → queue src
  TimingMetric preproc;          // preproc element sink → src
  TimingMetric filterElement;    // tensor_filter element sink → src
  TimingMetric inference;        // tensor_filter internal latency (pure NPU Invoke)
  TimingMetric postproc;         // tensor_sink callback total time
  TimingMetric fullLatency;      // queue sink → end of post-processing

  // Per-frame timings populated by the overlay's `frame-timing` signal.
  // decode/materialize are emitted from the tensor chain inside the
  // overlay; draw is emitted from the video chain. Each value covers
  // exactly one frame's work, so the averages are directly addable.
  TimingMetric decode;
  TimingMetric materialize;
  TimingMetric draw;

  // Per-buffer probe state (set by upstream/downstream-thread probes)
  struct timeval queueSinkStart;
  struct timeval queueSrcEnd;
  struct timeval preprocStart;
  struct timeval preprocEnd;
  struct timeval filterSinkStart;
  struct timeval filterSrcEnd;
  // Snapshot of queueSinkStart taken on the downstream thread inside
  // the queue src probe — race-safe for full-latency measurement.
  struct timeval currentFrameQueueStart;

  // tensor_filter element kept for queryInferenceLatency()
  GstElement *tensorFilter;

  void reset();

  /**
   * @brief Install pad probes on the named elements.
   *
   * Each name is the gst element `name=` value used in the pipeline
   * string. Any name may be NULL to skip that element's probes (useful
   * for binaries that don't have a thread queue or use a different
   * structure).
   *
   * @param pipeline    Top-level GstPipeline
   * @param queueName   queue element name (typically "thread-nn"), or NULL
   * @param preprocName edgefirstcameraadaptor element name (typically "preproc"), or NULL
   * @param filterName  tensor_filter element name (typically "tfilter"), or NULL
   * @return true on success
   */
  bool install(GstElement *pipeline,
               const char *queueName,
               const char *preprocName,
               const char *filterName);

  /**
   * @brief Wire the edgefirstoverlay `frame-timing` signal into decode /
   * materialize / draw metrics.
   *
   * Sets the overlay's `expose-timing` property to TRUE and connects the
   * signal so per-frame measurements flow into the corresponding
   * TimingMetric fields. Safe to call before the pipeline reaches PLAYING.
   *
   * @param overlay edgefirstoverlay element pointer (gst_bin_get_by_name).
   *                Caller retains ownership.
   * @return true if the signal connected
   */
  bool installOverlayTiming(GstElement *overlay);

  /** Release the cached tensor_filter reference. */
  void teardown();

  /**
   * @brief Record post-processing time + full pipeline latency.
   *
   * Call this from the tensor_sink new-data callback after the post-
   * processing work has completed. It records the post duration and
   * also computes the per-buffer full pipeline latency from the
   * race-safe currentFrameQueueStart snapshot.
   */
  void recordPost(const struct timeval &postStart, const struct timeval &postEnd);

  /**
   * @brief Query and record the tensor_filter `latency` property.
   * Call this from the tensor_sink new-data callback alongside
   * recordPost(). Reports the pure NPU Invoke time independent of
   * gstreamer element overhead.
   */
  void recordInference();

  /**
   * @brief Print the unified per-element + full latency report.
   * @param title   Section title (e.g. "KINARA ARA-2 NPU + EDGEFIRST")
   */
  void printReport(const char *title) const;
};


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

/**
 * @brief Set LIBCAMERA_PIPELINES_MATCH_LIST and NEUTRON_ENABLE_ZERO_COPY.
 * @param neutron_zero_copy  Pass true for pipelines that use the NnsDmaBufInputPool
 *                           (i.e. yolov8n_imx95 with edgefirstcameraadaptor).
 *                           Pass false for all other pipelines.
 */
void setupImx95Environment(bool neutron_zero_copy);


#endif // YOLOV8_COMMON_HPP
