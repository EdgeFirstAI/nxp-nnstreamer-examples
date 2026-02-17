/**
 * Copyright 2025 NXP
 * Copyright 2026 EdgeFirst AI (Au-Zone Technologies)
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * YOLOv8n 640x640 Reference Demo — Standard NXP Pipeline with Instrumentation
 *
 * Unified baseline binary supporting both i.MX 95 (Neutron NPU) and
 * i.MX 8M Plus (VeriSilicon NPU) via --platform flag.
 *
 * Purpose: Benchmarking baseline for comparison against EdgeFirst builds.
 * Uses the standard NXP 6-element preprocessing pipeline with manual
 * dequantization and per-class NMS. No EdgeFirst dependencies.
 *
 * PREPROCESSING PIPELINE:
 *   1. G2D Scale+Colorspace: imxvideoconvert_g2d (1920x1080 NV12 → 640x360 RGBA)
 *   2. Letterbox Padding: videobox (640x360 → 640x640 with black borders)
 *   3. Colorspace: videoconvert (RGBA → RGB, zero-copy metadata change)
 *   4. Tensor Conversion: tensor_converter (RGB frame → tensor buffer)
 *   5a. Tensor Transform #1: typecast uint8→int16 + add -128
 *   5b. Tensor Transform #2: typecast int16→int8
 *
 * INFERENCE: tensor_filter with tensorflow-lite + NPU delegate
 *
 * POST-PROCESSING: Manual INT8 dequantization + per-class NMS (nms.cpp)
 *
 * END-TO-END: PTS-correlated latency measurement from G2D input to
 * post-processing complete, with validation against sum of stages.
 */

#include <gst/gst.h>
#include <glib.h>
#include <glib-unix.h>
#include <cairo.h>
#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <sys/time.h>
#include <vector>

#include "logging.hpp"
#include "nms.hpp"

#include "common/yolov8_common.hpp"


/* ─── Platform configuration ──────────────────────────────────────── */

enum Platform { PLATFORM_IMX95, PLATFORM_IMX8MP };

struct PlatformConfig {
  const char *name;
  const char *npuName;
  const char *delegateLib;
  const char *defaultCamera;
  bool usesLibcamerasrc;   // true=libcamerasrc, false=v4l2src
};

static const PlatformConfig platformConfigs[] = {
  [PLATFORM_IMX95] = {
    .name = "i.MX 95",
    .npuName = "Neutron NPU",
    .delegateLib = "libneutron_delegate.so",
    .defaultCamera = NULL,  // libcamerasrc has no device parameter
    .usesLibcamerasrc = true,
  },
  [PLATFORM_IMX8MP] = {
    .name = "i.MX 8M Plus",
    .npuName = "VSI NPU",
    .delegateLib = "libvx_delegate.so",
    .defaultCamera = "/dev/video3",
    .usesLibcamerasrc = false,
  },
};


/* ─── Timing statistics ───────────────────────────────────────────── */

typedef struct {
  // Preprocessing stages (measured via pad probes)
  TimingMetric g2dScale;       // G2D resize + NV12→RGBA (FUSED in hardware)
  TimingMetric letterbox;      // videobox padding
  TimingMetric colorconv;      // RGBA → RGB conversion
  TimingMetric tensorConv;     // tensor_converter (RGB → tensor)
  TimingMetric tensorShift1;   // tensor_transform #1 (uint8→int16 + add -128)
  TimingMetric tensorShift2;   // tensor_transform #2 (int16→int8)
  TimingMetric preprocTotal;   // Total preprocessing

  // Inference
  TimingMetric inference;      // NPU execution (from tensor_filter)

  // Post-processing stages
  TimingMetric outputMmap;     // gst_memory_map for output tensor
  TimingMetric dequantExtract; // Dequantization + box extraction
  TimingMetric nms;            // Non-maximum suppression
  TimingMetric postprocTotal;  // Total post-processing

  // Rendering
  TimingMetric cairoDraw;      // Cairo rendering

  // End-to-end
  TimingMetric e2e;            // Frame-to-frame interval (throughput)
  TimingMetric e2ePipeline;    // True E2E latency (PTS-correlated)

  // Timestamps for pad probe timing
  struct timeval g2dStart;
  struct timeval g2dEnd;
  struct timeval letterboxEnd;
  struct timeval colorconvEnd;
  struct timeval tensorconvEnd;
  struct timeval tshift1End;
  struct timeval tshift2End;
  struct timeval lastFrameTime;
  bool firstFrame;

  // PTS-correlated E2E tracking
  std::map<GstClockTime, struct timeval> nnPipelineStart;
  GMutex e2eMutex;

  // Detection statistics
  int totalDetections;
  int framesWithDetections;
  int preNmsDetections;
} TimingStats;


/* ─── Application data ────────────────────────────────────────────── */

typedef struct {
  GstElement *gstPipeline;
  GMainLoop *loop;
  GstBus *bus;
  gboolean playing;
  std::vector<DetectedObject> results;
  std::vector<std::string> className;
  TimingStats timing;
  GstElement *tensorFilter;

  LetterboxParams letterbox;
  Platform platform;
  bool headless;
  bool videoLoop;   // Loop video on EOS
  double videoRate; // Playback rate (1.0 = normal, 0.5 = half speed)
  bool startedOnce; // Suppress repeated state change messages after initial startup
} AppData;


/* ─── Pad probes for preprocessing timing ─────────────────────────── */

static GstPadProbeReturn g2dSinkProbe(GstPad *, GstPadProbeInfo *info, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->timing.g2dStart, NULL);

  // Store timestamp keyed by PTS for E2E correlation
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (buffer) {
    GstClockTime pts = GST_BUFFER_PTS(buffer);
    if (GST_CLOCK_TIME_IS_VALID(pts)) {
      g_mutex_lock(&app->timing.e2eMutex);
      app->timing.nnPipelineStart[pts] = app->timing.g2dStart;
      while (app->timing.nnPipelineStart.size() > 100)
        app->timing.nnPipelineStart.erase(app->timing.nnPipelineStart.begin());
      g_mutex_unlock(&app->timing.e2eMutex);
    }
  }
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn g2dSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->timing.g2dEnd, NULL);
  app->timing.g2dScale.record(timeDiffMs(app->timing.g2dStart, app->timing.g2dEnd));
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn letterboxSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->timing.letterboxEnd, NULL);
  app->timing.letterbox.record(timeDiffMs(app->timing.g2dEnd, app->timing.letterboxEnd));
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn colorconvSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->timing.colorconvEnd, NULL);
  app->timing.colorconv.record(timeDiffMs(app->timing.letterboxEnd, app->timing.colorconvEnd));
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn tensorconvSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->timing.tensorconvEnd, NULL);
  app->timing.tensorConv.record(timeDiffMs(app->timing.colorconvEnd, app->timing.tensorconvEnd));
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn tshift1SrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->timing.tshift1End, NULL);
  app->timing.tensorShift1.record(timeDiffMs(app->timing.tensorconvEnd, app->timing.tshift1End));
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn tshift2SrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->timing.tshift2End, NULL);
  app->timing.tensorShift2.record(timeDiffMs(app->timing.tshift1End, app->timing.tshift2End));
  app->timing.preprocTotal.record(timeDiffMs(app->timing.g2dStart, app->timing.tshift2End));
  return GST_PAD_PROBE_OK;
}


/* ─── Timing report ───────────────────────────────────────────────── */

static void printTimingStatistics(AppData *app)
{
  const PlatformConfig &plat = platformConfigs[app->platform];

  printf("\n");
  printf("==============================================================================\n");
  printf("  REFERENCE PIPELINE — %s (%s)\n", plat.name, plat.npuName);
  printf("==============================================================================\n");

  printf("\n  PREPROCESSING (1920x1080 NV12 -> 640x640 INT8 tensor)\n");
  printf("  --------------------------------------------------------------------------\n");
  printMetric("1. G2D Scale + Colorspace [FUSED - HW accelerated]",
              "1920x1080 NV12 -> 640x360 RGBA", app->timing.g2dScale);
  printMetric("2. Letterbox Padding [videobox]",
              "640x360 -> 640x640 with black borders", app->timing.letterbox);
  printMetric("3. Colorspace Conversion [videoconvert]",
              "RGBA -> RGB for model input", app->timing.colorconv);
  printMetric("4. Tensor Conversion [tensor_converter]",
              "RGB frame -> tensor buffer", app->timing.tensorConv);
  printMetric("5a. Tensor Transform #1 [typecast + add]",
              "uint8->int16 + add -128", app->timing.tensorShift1);
  printMetric("5b. Tensor Transform #2 [typecast]",
              "int16->int8", app->timing.tensorShift2);

  if (app->timing.preprocTotal.count > 0) {
    printf("\n  >> PREPROCESSING TOTAL\n");
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms\n",
           app->timing.preprocTotal.avg(),
           app->timing.preprocTotal.minMs,
           app->timing.preprocTotal.maxMs);
  }

  printf("\n==============================================================================\n");
  printf("\n  INFERENCE (%s)\n", plat.npuName);
  printf("  --------------------------------------------------------------------------\n");
  if (app->timing.inference.count > 0) {
    printf("     tensor_filter with %s (10-frame moving average)\n", plat.delegateLib);
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d samples]\n",
           app->timing.inference.avg(), app->timing.inference.minMs,
           app->timing.inference.maxMs, app->timing.inference.count);
  }

  printf("\n==============================================================================\n");
  printf("\n  POST-PROCESSING (output tensor -> detection boxes)\n");
  printf("  --------------------------------------------------------------------------\n");
  printMetric("1. Output Tensor Read [gst_memory_map]",
              "mmap output buffer for CPU access", app->timing.outputMmap);
  printMetric("2. Dequantization + Box Extraction",
              "INT8->float, 8400 boxes x 84 channels, coordinate mapping",
              app->timing.dequantExtract);

  if (app->timing.nms.count > 0) {
    printf("  3. Non-Maximum Suppression [per-class NMS]\n");
    printf("     (IoU threshold: %.2f, confidence threshold: %.2f)\n",
           NMS_IOU_THRESHOLD, CONF_THRESHOLD);
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d frames]\n",
           app->timing.nms.avg(), app->timing.nms.minMs,
           app->timing.nms.maxMs, app->timing.nms.count);
  }

  if (app->timing.postprocTotal.count > 0) {
    printf("\n  >> POST-PROCESSING TOTAL\n");
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms\n",
           app->timing.postprocTotal.avg(),
           app->timing.postprocTotal.minMs,
           app->timing.postprocTotal.maxMs);
  }

  if (!app->headless && app->timing.cairoDraw.count > 0) {
    printf("\n==============================================================================\n");
    printf("\n  RENDERING\n");
    printf("  --------------------------------------------------------------------------\n");
    printMetric("1. Cairo Drawing [bounding boxes + labels]", NULL, app->timing.cairoDraw);
  }

  printf("\n==============================================================================\n");
  printf("\n  END-TO-END TIMING ANALYSIS\n");
  printf("  --------------------------------------------------------------------------\n");

  double sumPreproc = app->timing.preprocTotal.avg();
  double sumInference = app->timing.inference.avg();
  double sumPostproc = app->timing.postprocTotal.avg();
  double sumRender = app->timing.cairoDraw.avg();
  double sumNN = sumPreproc + sumInference + sumPostproc;
  double sumTotal = sumNN + sumRender;

  if (app->timing.e2ePipeline.count > 0) {
    printf("\n  1. E2E NN Pipeline (PTS-correlated: g2d sink -> post-processing)\n");
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d frames]\n",
           app->timing.e2ePipeline.avg(), app->timing.e2ePipeline.minMs,
           app->timing.e2ePipeline.maxMs, app->timing.e2ePipeline.count);
  }

  printf("\n  2. Sum of Stages\n");
  printf("     +---------------------------------------------------------------+\n");
  printf("     | Preprocessing (G2D->tensorShift2):          %7.3f ms          |\n", sumPreproc);
  printf("     | Inference (%s):                     %7.3f ms          |\n", plat.npuName, sumInference);
  printf("     | Post-processing (mmap->NMS):              %7.3f ms          |\n", sumPostproc);
  if (!app->headless)
    printf("     | Rendering (Cairo draw):                   %7.3f ms          |\n", sumRender);
  printf("     +---------------------------------------------------------------+\n");
  printf("     | NN Branch Total (preproc+inf+postproc):   %7.3f ms          |\n", sumNN);
  printf("     | Full Pipeline Sum:                        %7.3f ms          |\n", sumTotal);
  printf("     +---------------------------------------------------------------+\n");

  if (app->timing.e2ePipeline.count > 0) {
    double e2eAvg = app->timing.e2ePipeline.avg();
    double coverage = (sumNN / e2eAvg) * 100.0;
    printf("\n  3. Timing Validation\n");
    printf("     E2E NN Pipeline: %7.3f ms  |  Sum of NN Stages: %7.3f ms\n", e2eAvg, sumNN);
    printf("     Coverage: %6.1f%%\n", coverage);
  }

  if (app->timing.e2e.count > 0) {
    double avgMs = app->timing.e2e.avg();
    printf("\n  4. Frame Throughput\n");
    printf("     Average: %7.3f ms (%5.1f FPS)  |  Min: %7.3f ms  |  Max: %7.3f ms\n",
           avgMs, 1000.0 / avgMs, app->timing.e2e.minMs, app->timing.e2e.maxMs);
    printf("     Frames: %d\n", app->timing.e2e.count);
  }

  printf("\n==============================================================================\n");
  printf("\n  DETECTION STATISTICS\n");
  printf("  --------------------------------------------------------------------------\n");
  printf("     Pre-NMS candidates:      %6d  (avg %.1f/frame)\n",
         app->timing.preNmsDetections,
         app->timing.postprocTotal.count > 0
             ? (double)app->timing.preNmsDetections / app->timing.postprocTotal.count : 0);
  printf("     Post-NMS detections:     %6d  (avg %.1f/frame)\n",
         app->timing.totalDetections,
         app->timing.framesWithDetections > 0
             ? (double)app->timing.totalDetections / app->timing.framesWithDetections : 0);
  printf("     Frames with detections:  %6d / %d\n",
         app->timing.framesWithDetections, app->timing.postprocTotal.count);
  printf("\n==============================================================================\n");
}


/* ─── GStreamer callbacks ─────────────────────────────────────────── */

static gboolean sigintHandler(gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  log_info("SIGINT — stopping.\n");
  printTimingStatistics(app);
  g_main_loop_quit(app->loop);
  return TRUE;
}

static void busCallback(GstBus *, GstMessage *message, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  switch (GST_MESSAGE_TYPE(message)) {
    case GST_MESSAGE_ERROR: {
      GError *err;
      gchar *debugInfo;
      gst_message_parse_error(message, &err, &debugInfo);
      log_error("Error from %s: %s\n", GST_OBJECT_NAME(message->src), err->message);
      log_debug("Debug: %s\n", debugInfo ? debugInfo : "none");
      g_error_free(err);
      g_free(debugInfo);
      g_main_loop_quit(app->loop);
      break;
    }
    case GST_MESSAGE_EOS:
      if (app->videoLoop) {
        gst_element_seek(app->gstPipeline, app->videoRate, GST_FORMAT_TIME,
                         (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT),
                         GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, 0);
      } else {
        log_info("End-Of-Stream reached.\n");
        printTimingStatistics(app);
        g_main_loop_quit(app->loop);
      }
      break;
    case GST_MESSAGE_STATE_CHANGED: {
      GstState oldState, newState, pendingState;
      gst_message_parse_state_changed(message, &oldState, &newState, &pendingState);
      if (GST_MESSAGE_SRC(message) == GST_OBJECT(app->gstPipeline) &&
          oldState != newState) {
        if (!app->startedOnce) {
          log_info("Pipeline state: %s -> %s\n",
                   gst_element_state_get_name(oldState),
                   gst_element_state_get_name(newState));
        }
        if (newState == GST_STATE_PLAYING) {
          app->playing = true;
          if (!app->startedOnce && app->videoRate != 1.0) {
            gst_element_seek(app->gstPipeline, app->videoRate, GST_FORMAT_TIME,
                             (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT),
                             GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, 0);
          }
          app->startedOnce = true;
        } else {
          app->playing = false;
        }
      }
      break;
    }
    default:
      break;
  }
}

static void newDataCallback(GstElement *, GstBuffer *buffer, gpointer user_data)
{
  struct timeval callbackStart, mmapEnd, dequantEnd, nmsEnd;
  gettimeofday(&callbackStart, NULL);

  AppData *app = (AppData *)user_data;

  // Frame-to-frame interval
  if (!app->timing.firstFrame) {
    app->timing.e2e.record(timeDiffMs(app->timing.lastFrameTime, callbackStart));
  }
  app->timing.lastFrameTime = callbackStart;
  app->timing.firstFrame = false;

  // Inference latency from tensor_filter
  if (app->tensorFilter) {
    gint64 latencyUs = 0;
    g_object_get(app->tensorFilter, "latency", &latencyUs, NULL);
    if (latencyUs > 0)
      app->timing.inference.record(latencyUs / 1000.0);
  }

  // Validate buffer
  if (!GST_IS_BUFFER(buffer)) { log_error("Invalid buffer\n"); return; }
  if (gst_buffer_n_memory(buffer) != 1) { log_error("Expected 1 tensor\n"); return; }

  // Map output tensor
  GstMemory *mem = gst_buffer_peek_memory(buffer, 0);
  GstMapInfo info;
  if (!gst_memory_map(mem, &info, GST_MAP_READ)) {
    log_error("Can't map output tensor\n");
    return;
  }
  int8_t *outputTensor = reinterpret_cast<int8_t *>(info.data);

  gettimeofday(&mmapEnd, NULL);
  app->timing.outputMmap.record(timeDiffMs(callbackStart, mmapEnd));

  // Dequantization + box extraction
  std::vector<DetectedObject> output;
  for (int bIdx = 0; bIdx < NUM_TOTAL_BOXES; ++bIdx) {
    int maxClassConfVal = -128;
    int maxClassIdx = -1;
    for (int cIdx = NUM_COORDINATES; cIdx < MODEL_OUTPUT_WIDTH; cIdx++) {
      int val = outputTensor[cIdx * NUM_TOTAL_BOXES + bIdx];
      if (val > maxClassConfVal) {
        maxClassConfVal = val;
        maxClassIdx = cIdx;
      }
    }
    if (maxClassConfVal > QUANTIZED_THRESHOLD) {
      int8_t raw_cx = outputTensor[0 * NUM_TOTAL_BOXES + bIdx];
      int8_t raw_cy = outputTensor[1 * NUM_TOTAL_BOXES + bIdx];
      int8_t raw_w  = outputTensor[2 * NUM_TOTAL_BOXES + bIdx];
      int8_t raw_h  = outputTensor[3 * NUM_TOTAL_BOXES + bIdx];

      float cx = (raw_cx - ZERO_POINT) * SCALE_FACTOR;
      float cy = (raw_cy - ZERO_POINT) * SCALE_FACTOR;
      float w  = (raw_w  - ZERO_POINT) * SCALE_FACTOR;
      float h  = (raw_h  - ZERO_POINT) * SCALE_FACTOR;

      // Model output normalized [0,1] → pixel coords → remove letterbox → scale to source
      float px = cx * MODEL_INPUT_SIZE - app->letterbox.padX;
      float py = cy * MODEL_INPUT_SIZE - app->letterbox.padY;
      float pw = w * MODEL_INPUT_SIZE;
      float ph = h * MODEL_INPUT_SIZE;
      px /= app->letterbox.scale;
      py /= app->letterbox.scale;
      pw /= app->letterbox.scale;
      ph /= app->letterbox.scale;

      DetectedObject object;
      object.x = static_cast<int>(std::max(0.f, px - pw / 2.f));
      object.y = static_cast<int>(std::max(0.f, py - ph / 2.f));
      object.width = static_cast<int>(std::min((float)SOURCE_WIDTH, pw));
      object.height = static_cast<int>(std::min((float)SOURCE_HEIGHT, ph));

      if (object.width < 5 || object.height < 5) continue;
      if (object.x + object.width > SOURCE_WIDTH || object.y + object.height > SOURCE_HEIGHT) continue;

      object.classId = maxClassIdx - NUM_COORDINATES;
      object.prob = (maxClassConfVal - ZERO_POINT) * SCALE_FACTOR;
      object.valid = true;
      output.push_back(object);
    }
  }

  gst_memory_unmap(mem, &info);

  gettimeofday(&dequantEnd, NULL);
  app->timing.dequantExtract.record(timeDiffMs(mmapEnd, dequantEnd));
  app->timing.preNmsDetections += output.size();

  // Per-class NMS
  std::sort(output.begin(), output.end(),
            [](const DetectedObject &a, const DetectedObject &b) { return a.prob > b.prob; });

  app->results.clear();
  std::map<int, std::vector<DetectedObject>> perClassBoxes;
  for (auto &obj : output)
    perClassBoxes[obj.classId].push_back(obj);

  for (auto &[classId, boxes] : perClassBoxes) {
    for (size_t i = 0; i < boxes.size(); ++i) {
      if (!boxes[i].valid) continue;
      app->results.push_back(boxes[i]);
      for (size_t j = i + 1; j < boxes.size(); ++j) {
        if (boxes[j].valid && iou(boxes[i], boxes[j]) > NMS_IOU_THRESHOLD)
          boxes[j].valid = false;
      }
    }
  }

  gettimeofday(&nmsEnd, NULL);
  app->timing.nms.record(timeDiffMs(dequantEnd, nmsEnd));

  if (!app->results.empty()) {
    app->timing.framesWithDetections++;
    app->timing.totalDetections += app->results.size();
  }

  app->timing.postprocTotal.record(timeDiffMs(callbackStart, nmsEnd));

  // PTS-correlated E2E
  GstClockTime pts = GST_BUFFER_PTS(buffer);
  if (GST_CLOCK_TIME_IS_VALID(pts)) {
    g_mutex_lock(&app->timing.e2eMutex);
    auto it = app->timing.nnPipelineStart.find(pts);
    if (it != app->timing.nnPipelineStart.end()) {
      app->timing.e2ePipeline.record(timeDiffMs(it->second, nmsEnd));
      app->timing.nnPipelineStart.erase(it);
    }
    g_mutex_unlock(&app->timing.e2eMutex);
  }
}

static void drawCallback(GstElement *, cairo_t *cr, guint64, guint64, gpointer user_data)
{
  AppData *app = (AppData *)user_data;

  struct timeval startTime, endTime;
  gettimeofday(&startTime, NULL);

  // Draw detection boxes with class name and confidence
  cairo_select_font_face(cr, "monospace", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);

  for (auto &det : app->results) {
    char label[64];
    snprintf(label, sizeof(label), "%s %.0f%%",
             app->className.at(det.classId).c_str(), det.prob * 100.0f);

    // Green box with 2px border
    cairo_set_source_rgb(cr, 0, 1, 0);
    cairo_set_line_width(cr, 2.0);
    cairo_rectangle(cr, det.x, det.y, det.width, det.height);
    cairo_stroke(cr);

    // Label background
    cairo_set_font_size(cr, 14);
    cairo_text_extents_t extents;
    cairo_text_extents(cr, label, &extents);
    cairo_set_source_rgba(cr, 0, 0, 0, 0.6);
    cairo_rectangle(cr, det.x, det.y - 18, extents.width + 8, 18);
    cairo_fill(cr);

    // Label text
    cairo_set_source_rgb(cr, 0, 1, 0);
    cairo_move_to(cr, det.x + 4, det.y - 4);
    cairo_show_text(cr, label);
  }

  // Draw timing overlay in top-left corner
  double fps = app->timing.e2e.count > 0 ? 1000.0 / app->timing.e2e.avg() : 0;
  double infMs = app->timing.inference.avg();
  double prepMs = app->timing.preprocTotal.avg();
  int nDet = (int)app->results.size();

  char overlay[256];
  snprintf(overlay, sizeof(overlay),
           "%.1f FPS | Inf: %.1f ms | Pre: %.1f ms | %d det",
           fps, infMs, prepMs, nDet);

  // Background bar
  cairo_set_font_size(cr, 16);
  cairo_text_extents_t ov_ext;
  cairo_text_extents(cr, overlay, &ov_ext);
  cairo_set_source_rgba(cr, 0, 0, 0, 0.7);
  cairo_rectangle(cr, 0, 0, ov_ext.width + 20, 28);
  cairo_fill(cr);

  // Overlay text
  cairo_set_source_rgb(cr, 1, 1, 1);
  cairo_move_to(cr, 10, 20);
  cairo_show_text(cr, overlay);

  gettimeofday(&endTime, NULL);
  app->timing.cairoDraw.record(timeDiffMs(startTime, endTime));
}


/* ─── Pipeline construction ───────────────────────────────────────── */

static char *buildPipeline(Platform platform, const std::string &model,
                           const std::string &camera, const std::string &video,
                           const std::string &image, const LetterboxParams &lb, bool headless)
{
  const PlatformConfig &plat = platformConfigs[platform];

  // Source element
  std::string source;
  if (!image.empty()) {
    // Static image input — use imagefreeze to create continuous stream
    char *s;
    if (headless) {
      s = g_strdup_printf(
          "filesrc location=%s ! jpegdec ! imxvideoconvert_g2d ! "
          "video/x-raw,width=%d,height=%d ! imagefreeze ! "
          "queue name=thread-nn leaky=2 max-size-buffers=2",
          image.c_str(), SOURCE_WIDTH, SOURCE_HEIGHT);
    } else {
      s = g_strdup_printf(
          "filesrc location=%s ! jpegdec ! imxvideoconvert_g2d ! "
          "video/x-raw,width=%d,height=%d ! imagefreeze ! "
          "tee name=t "
          "t. ! queue name=thread-nn leaky=2 max-size-buffers=2",
          image.c_str(), SOURCE_WIDTH, SOURCE_HEIGHT);
    }
    source = s;
    g_free(s);
  } else if (!video.empty()) {
    // Video file input — use tee for display branch unless headless
    char *s;
    if (headless) {
      s = g_strdup_printf(
          "filesrc location=%s ! qtdemux ! h264parse ! v4l2h264dec ! "
          "queue name=thread-nn leaky=2 max-size-buffers=2",
          video.c_str());
    } else {
      s = g_strdup_printf(
          "filesrc location=%s ! qtdemux ! h264parse ! v4l2h264dec ! "
          "tee name=t "
          "t. ! queue name=thread-nn leaky=2 max-size-buffers=2",
          video.c_str());
    }
    source = s;
    g_free(s);
  } else if (plat.usesLibcamerasrc) {
    // i.MX 95 camera
    char *s = g_strdup_printf(
        "libcamerasrc ! video/x-raw,format=NV12,width=%d,height=%d ! "
        "tee name=t "
        "t. ! queue name=thread-nn leaky=2 max-size-buffers=2",
        SOURCE_WIDTH, SOURCE_HEIGHT);
    source = s;
    g_free(s);
  } else {
    // i.MX 8M Plus camera
    char *s = g_strdup_printf(
        "v4l2src device=%s ! video/x-raw,format=NV12,width=%d,height=%d,framerate=30/1 ! "
        "tee name=t "
        "t. ! queue name=thread-nn leaky=2 max-size-buffers=2",
        camera.c_str(), SOURCE_WIDTH, SOURCE_HEIGHT);
    source = s;
    g_free(s);
  }

  // NN branch: preprocessing + inference + sink
  char *nnBranch = g_strdup_printf(
      "%s ! "
      "imxvideoconvert_g2d name=g2d_scale ! video/x-raw,width=%d,height=%d,format=RGBA ! "
      "videobox name=letterbox left=%d right=%d top=%d bottom=%d fill=black ! "
      "videoconvert name=colorconv ! video/x-raw,format=RGB ! "
      "tensor_converter name=tconv ! "
      "tensor_transform name=tshift1 mode=arithmetic option=typecast:int16,add:-128 ! "
      "tensor_transform name=tshift2 mode=typecast option=int8 ! "
      "tensor_filter name=tfilter framework=tensorflow-lite model=%s "
      "custom=Delegate:External,ExtDelegateLib:%s latency=1 ! "
      "tensor_sink name=inferenceOutput",
      source.c_str(),
      lb.scaledW, lb.scaledH,
      -lb.padX, -lb.padRight, -lb.padY, -lb.padBottom,
      model.c_str(), plat.delegateLib);

  char *pipeline;
  if (headless) {
    // Headless: no display branch
    pipeline = nnBranch;
  } else {
    // With display: add display branch via tee
    pipeline = g_strdup_printf(
        "%s "
        "t. ! queue name=thread-img max-size-buffers=2 ! "
        "imxvideoconvert_g2d ! "
        "cairooverlay name=cairo ! "
        "waylandsink sync=%s",
        nnBranch, (!video.empty()) ? "true" : "false");
    g_free(nnBranch);
  }

  return pipeline;
}


/* ─── Install pad probes ──────────────────────────────────────────── */

static void installProbes(GstElement *pipeline, AppData *app)
{
  struct ProbeInfo {
    const char *elementName;
    const char *padName;
    GstPadProbeCallback callback;
  };

  ProbeInfo probes[] = {
    {"g2d_scale", "sink", g2dSinkProbe},
    {"g2d_scale", "src",  g2dSrcProbe},
    {"letterbox", "src",  letterboxSrcProbe},
    {"colorconv", "src",  colorconvSrcProbe},
    {"tconv",     "src",  tensorconvSrcProbe},
    {"tshift1",   "src",  tshift1SrcProbe},
    {"tshift2",   "src",  tshift2SrcProbe},
  };

  for (auto &p : probes) {
    GstElement *elem = gst_bin_get_by_name(GST_BIN(pipeline), p.elementName);
    if (elem) {
      GstPad *pad = gst_element_get_static_pad(elem, p.padName);
      if (pad) {
        gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, p.callback, app, NULL);
        gst_object_unref(pad);
      }
      gst_object_unref(elem);
    }
  }
}


/* ─── Command line parsing ────────────────────────────────────────── */

static int parseArgs(int argc, char **argv, std::string &model, std::string &camera,
                     std::string &video, std::string &image, Platform &platform, bool &headless,
                     int &numFrames, double &speed)
{
  static struct option longOptions[] = {
    {"help",     no_argument,       0, 'h'},
    {"model",    required_argument, 0, 'm'},
    {"camera",   required_argument, 0, 'c'},
    {"video",    required_argument, 0, 'v'},
    {"image",    required_argument, 0, 'i'},
    {"headless", no_argument,       0, 'H'},
    {"frames",   required_argument, 0, 'n'},
    {"platform", required_argument, 0, 'p'},
    {"speed",    required_argument, 0, 's'},
    {0, 0, 0, 0}
  };

  int c;
  while ((c = getopt_long(argc, argv, "hm:c:v:i:Hn:p:s:", longOptions, NULL)) != -1) {
    switch (c) {
      case 'h':
        std::cout
            << "YOLOv8n 640x640 Reference Pipeline (standard NXP preprocessing)\n\n"
            << "Usage: " << argv[0] << " -m MODEL --platform imx95|imx8mp [options]\n\n"
            << "Options:\n"
            << "  -m, --model PATH      Model file (.tflite) [required]\n"
            << "  -p, --platform NAME   Platform: imx95 or imx8mp [required]\n"
            << "  -c, --camera DEVICE   Camera device (default: platform-specific)\n"
            << "  -v, --video FILE      Video file input (H.264 MP4)\n"
            << "  -i, --image FILE      Static image input (JPEG)\n"
            << "  -s, --speed RATE      Video playback speed (0.25=quarter, 0.5=half, 1.0=normal)\n"
            << "  -H, --headless        No display output\n"
            << "  -n, --frames N        Stop after N frames (0=infinite, default=0)\n"
            << "  -h, --help            Show this help\n";
        return 1;
      case 'm': model = optarg; break;
      case 'c': camera = optarg; break;
      case 'v': video = optarg; break;
      case 'i': image = optarg; break;
      case 'H': headless = true; break;
      case 'n': numFrames = atoi(optarg); break;
      case 's': speed = atof(optarg); break;
      case 'p':
        if (strcmp(optarg, "imx95") == 0)
          platform = PLATFORM_IMX95;
        else if (strcmp(optarg, "imx8mp") == 0)
          platform = PLATFORM_IMX8MP;
        else {
          std::cerr << "Unknown platform: " << optarg << " (use imx95 or imx8mp)\n";
          return -1;
        }
        break;
    }
  }
  return 0;
}


/* ─── Main ────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
  std::string model, camera, video, image;
  Platform platform = PLATFORM_IMX8MP;
  bool headless = false;
  int numFrames = 0;
  double speed = 1.0;
  bool platformSet = false;

  // Check if --platform was provided
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--platform") == 0 || strcmp(argv[i], "-p") == 0)
      platformSet = true;
  }

  int ret = parseArgs(argc, argv, model, camera, video, image, platform, headless, numFrames, speed);
  if (ret != 0) return ret > 0 ? 0 : 1;

  if (model.empty()) {
    log_error("Provide model path with -m\n");
    return 1;
  }
  if (!platformSet) {
    log_error("Provide platform with --platform imx95 or --platform imx8mp\n");
    return 1;
  }

  const PlatformConfig &plat = platformConfigs[platform];

  // Set default camera if not specified
  if (camera.empty() && plat.defaultCamera)
    camera = plat.defaultCamera;

  // Platform-specific environment setup
  if (platform == PLATFORM_IMX95) {
    const char *pm = getenv("LIBCAMERA_PIPELINES_MATCH_LIST");
    if (!pm || strlen(pm) == 0) {
      log_info("Setting LIBCAMERA_PIPELINES_MATCH_LIST='nxp/neo,imx8-isi'\n");
      setenv("LIBCAMERA_PIPELINES_MATCH_LIST", "nxp/neo,imx8-isi", 1);
    }
    const char *zc = getenv("NEUTRON_ENABLE_ZERO_COPY");
    if (!zc) {
      log_info("Setting NEUTRON_ENABLE_ZERO_COPY=0\n");
      setenv("NEUTRON_ENABLE_ZERO_COPY", "0", 1);
    }
  }

  log_info("YOLOv8n 640x640 Reference Pipeline — %s (%s)\n", plat.name, plat.npuName);
  log_info("Model: %s\n", model.c_str());
  log_info("Delegate: %s\n", plat.delegateLib);
  if (!image.empty()) {
    log_info("Input: image (%s)\n", image.c_str());
  } else if (!video.empty()) {
    log_info("Input: video (%s)\n", video.c_str());
    if (speed != 1.0) {
      log_info("Playback speed: %.2fx\n", speed);
    }
  } else {
    log_info("Input: camera (%s)\n", camera.empty() ? "libcamerasrc" : camera.c_str());
  }
  if (headless) {
    log_info("Mode: headless (no display)\n");
  }

  gst_init(&argc, &argv);

  // Calculate letterbox
  LetterboxParams lb = calculateLetterbox(SOURCE_WIDTH, SOURCE_HEIGHT);
  log_info("Letterbox: %dx%d -> scale %.4f -> %dx%d + pad L=%d R=%d T=%d B=%d\n",
           SOURCE_WIDTH, SOURCE_HEIGHT, lb.scale, lb.scaledW, lb.scaledH,
           lb.padX, lb.padRight, lb.padY, lb.padBottom);

  // Build pipeline
  char *pipelineStr = buildPipeline(platform, model, camera, video, image, lb, headless);
  log_info("Pipeline: %s\n\n", pipelineStr);

  // Initialize app data
  AppData app = {};
  app.className = getCocoClassNames();
  app.letterbox = lb;
  app.platform = platform;
  app.headless = headless;
  app.videoLoop = !video.empty() && image.empty();  // Don't loop for image input
  app.videoRate = speed;
  app.startedOnce = false;
  app.timing.g2dScale.reset();
  app.timing.letterbox.reset();
  app.timing.colorconv.reset();
  app.timing.tensorConv.reset();
  app.timing.tensorShift1.reset();
  app.timing.tensorShift2.reset();
  app.timing.preprocTotal.reset();
  app.timing.inference.reset();
  app.timing.outputMmap.reset();
  app.timing.dequantExtract.reset();
  app.timing.nms.reset();
  app.timing.postprocTotal.reset();
  app.timing.cairoDraw.reset();
  app.timing.e2e.reset();
  app.timing.e2ePipeline.reset();
  app.timing.firstFrame = true;
  g_mutex_init(&app.timing.e2eMutex);

  // Create pipeline
  app.loop = g_main_loop_new(NULL, FALSE);
  app.gstPipeline = gst_parse_launch(pipelineStr, NULL);
  g_free(pipelineStr);

  if (!app.gstPipeline) {
    log_error("Failed to create pipeline\n");
    return 1;
  }

  // Connect signals
  app.bus = gst_element_get_bus(app.gstPipeline);
  gst_bus_add_signal_watch(app.bus);
  g_signal_connect(app.bus, "message", G_CALLBACK(busCallback), &app);

  GstElement *tsink = gst_bin_get_by_name(GST_BIN(app.gstPipeline), "inferenceOutput");
  g_signal_connect(tsink, "new-data", G_CALLBACK(newDataCallback), &app);
  gst_object_unref(tsink);

  // Cairo overlay (only when displaying)
  if (!headless) {
    GstElement *cairo = gst_bin_get_by_name(GST_BIN(app.gstPipeline), "cairo");
    if (cairo) {
      g_signal_connect(cairo, "draw", G_CALLBACK(drawCallback), &app);
      gst_object_unref(cairo);
    }
  }

  app.tensorFilter = gst_bin_get_by_name(GST_BIN(app.gstPipeline), "tfilter");

  // Install timing probes
  installProbes(app.gstPipeline, &app);

  // SIGINT handler
  g_unix_signal_add(SIGINT, sigintHandler, &app);

  // Run
  gst_element_set_state(app.gstPipeline, GST_STATE_PLAYING);
  g_main_loop_run(app.loop);

  // Cleanup
  gst_element_set_state(app.gstPipeline, GST_STATE_NULL);
  if (app.tensorFilter)
    gst_object_unref(app.tensorFilter);
  gst_object_unref(app.bus);
  gst_object_unref(app.gstPipeline);
  g_main_loop_unref(app.loop);
  g_mutex_clear(&app.timing.e2eMutex);

  return 0;
}
