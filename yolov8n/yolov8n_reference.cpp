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

  // End-to-end (PTS-correlated)
  TimingMetric e2ePipeline;

  // Timestamps for pad probe timing
  struct timeval g2dStart;
  struct timeval g2dEnd;
  struct timeval letterboxEnd;
  struct timeval colorconvEnd;
  struct timeval tensorconvEnd;
  struct timeval tshift1End;
  struct timeval tshift2End;

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

  // Quantization parameters (from GstNnsTensorQuantMeta)
  double quantScale;
  int64_t quantZeroPoint;
  int quantizedThreshold;
  bool quantInitialized = false;

  // Common infrastructure
  BusCallbackCtx busCtx;
  ThroughputTracker throughput;
  PtsTracker ptsTracker;

  // Frame counting
  int numFrames;
  int frameCount;
  bool timingPrinted;
} AppData;


/* ─── Pad probes for preprocessing timing ─────────────────────────── */

static GstPadProbeReturn g2dSinkProbe(GstPad *, GstPadProbeInfo *info, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->timing.g2dStart, NULL);

  // Store timestamp keyed by PTS for E2E correlation
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  app->ptsTracker.recordStart(buffer, app->timing.g2dStart);
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

static void printTimingStatistics(void *userData)
{
  AppData *app = (AppData *)userData;
  if (app->timingPrinted) return;
  app->timingPrinted = true;
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

  if (app->throughput.metric.count > 0) {
    double avgMs = app->throughput.metric.avg();
    printf("\n  4. Frame Throughput\n");
    printf("     Average: %7.3f ms (%5.1f FPS)  |  Min: %7.3f ms  |  Max: %7.3f ms\n",
           avgMs, 1000.0 / avgMs, app->throughput.metric.minMs, app->throughput.metric.maxMs);
    printf("     Frames: %d\n", app->throughput.metric.count);
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

static void newDataCallback(GstElement *, GstBuffer *buffer, gpointer user_data)
{
  struct timeval callbackStart, mmapEnd, dequantEnd, nmsEnd;
  gettimeofday(&callbackStart, NULL);

  AppData *app = (AppData *)user_data;

  // Frame-to-frame interval
  app->throughput.tick(callbackStart);

  // Inference latency from tensor_filter
  queryInferenceLatency(app->tensorFilter, app->timing.inference);

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

  // Extract quantization parameters from GstNnsTensorQuantMeta on first frame
  if (!app->quantInitialized) {
    QuantParams qp;
    if (!extractQuantParams(buffer, 0, qp)) {
      log_error("No quant meta — requires updated NNStreamer\n");
      gst_memory_unmap(mem, &info);
      return;
    }
    app->quantScale = qp.scale;
    app->quantZeroPoint = qp.zeroPoint;
    app->quantizedThreshold =
        static_cast<int>((CONF_THRESHOLD / qp.scale) + qp.zeroPoint);
    log_info("Quant meta: scale=%g zero_point=%" G_GINT64_FORMAT "\n",
             qp.scale, qp.zeroPoint);
    app->quantInitialized = true;
  }

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
    if (maxClassConfVal > app->quantizedThreshold) {
      int8_t raw_cx = outputTensor[0 * NUM_TOTAL_BOXES + bIdx];
      int8_t raw_cy = outputTensor[1 * NUM_TOTAL_BOXES + bIdx];
      int8_t raw_w  = outputTensor[2 * NUM_TOTAL_BOXES + bIdx];
      int8_t raw_h  = outputTensor[3 * NUM_TOTAL_BOXES + bIdx];

      float cx = (raw_cx - app->quantZeroPoint) * app->quantScale;
      float cy = (raw_cy - app->quantZeroPoint) * app->quantScale;
      float w  = (raw_w  - app->quantZeroPoint) * app->quantScale;
      float h  = (raw_h  - app->quantZeroPoint) * app->quantScale;

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
      object.prob = (maxClassConfVal - app->quantZeroPoint) * app->quantScale;
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
  struct timeval ptsStart;
  if (app->ptsTracker.consumeStart(buffer, ptsStart)) {
    app->timing.e2ePipeline.record(timeDiffMs(ptsStart, nmsEnd));
  }

  // Frame counting for -n option
  if (app->numFrames > 0 && ++app->frameCount >= app->numFrames) {
    g_main_loop_quit(app->loop);
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
  double fps = app->throughput.metric.count > 0 ? 1000.0 / app->throughput.metric.avg() : 0;
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

static char *buildPipeline(Platform platform, const ParsedArgs &pargs,
                           const LetterboxParams &lb, bool headless)
{
  const PlatformConfig &plat = platformConfigs[platform];

  // Source element
  InputSource srcType = determineInputSource(pargs, plat.usesLibcamerasrc);
  char *srcStr = buildSourceElement(srcType, pargs);

  // For display mode, wrap source with tee (except headless)
  std::string source;
  if (headless || srcType == INPUT_IMAGE) {
    if (!headless && srcType == INPUT_IMAGE) {
      char *s = g_strdup_printf("%s ! tee name=t "
          "t. ! queue name=thread-nn leaky=2 max-size-buffers=2", srcStr);
      source = s;
      g_free(s);
    } else {
      char *s = g_strdup_printf("%s ! queue name=thread-nn leaky=2 max-size-buffers=2", srcStr);
      source = s;
      g_free(s);
    }
  } else {
    char *s = g_strdup_printf("%s ! tee name=t "
        "t. ! queue name=thread-nn leaky=2 max-size-buffers=2", srcStr);
    source = s;
    g_free(s);
  }
  g_free(srcStr);

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
      pargs.model.c_str(), plat.delegateLib);

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
        nnBranch, (!pargs.video.empty() || !pargs.image.empty()) ? "true" : "false");
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


/* ─── Main ────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
  ParsedArgs pargs;
  pargs.camera = "";  // Will be set from platform default

  uint32_t flags = ARG_MODEL | ARG_CAMERA | ARG_VIDEO | ARG_IMAGE |
                   ARG_HEADLESS | ARG_INSTRUMENTED | ARG_NUM_FRAMES | ARG_PLATFORM | ARG_SPEED;

  int ret = parseArgs(argc, argv, flags,
      "YOLOv8n 640x640 Reference Pipeline (standard NXP preprocessing)", pargs);
  if (ret != 0) return ret > 0 ? 0 : 1;

  if (pargs.model.empty()) {
    log_error("Provide model path with -m\n");
    return 1;
  }

  // Parse platform
  Platform platform = PLATFORM_IMX8MP;
  bool platformSet = false;
  if (!pargs.platformStr.empty()) {
    if (pargs.platformStr == "imx95") {
      platform = PLATFORM_IMX95;
      platformSet = true;
    } else if (pargs.platformStr == "imx8mp") {
      platform = PLATFORM_IMX8MP;
      platformSet = true;
    } else {
      log_error("Unknown platform: %s (use imx95 or imx8mp)\n", pargs.platformStr.c_str());
      return 1;
    }
  }

  if (!platformSet) {
    log_error("Provide platform with --platform imx95 or --platform imx8mp\n");
    return 1;
  }

  const PlatformConfig &plat = platformConfigs[platform];

  // Set default camera if not specified
  if (pargs.camera.empty() && plat.defaultCamera)
    pargs.camera = plat.defaultCamera;

  // Platform-specific environment setup
  if (platform == PLATFORM_IMX95)
    setupImx95Environment(false);

  log_info("YOLOv8n 640x640 Reference Pipeline — %s (%s)\n", plat.name, plat.npuName);
  log_info("Model: %s\n", pargs.model.c_str());
  log_info("Delegate: %s\n", plat.delegateLib);
  if (!pargs.image.empty()) {
    log_info("Input: image (%s)\n", pargs.image.c_str());
  } else if (!pargs.video.empty()) {
    log_info("Input: video (%s)\n", pargs.video.c_str());
    if (pargs.speed != 1.0) {
      log_info("Playback speed: %.2fx\n", pargs.speed);
    }
  } else {
    log_info("Input: camera (%s)\n", pargs.camera.empty() ? "libcamerasrc" : pargs.camera.c_str());
  }
  log_info("Mode: %s\n", pargs.headless ? "headless" : "display");

  gst_init(&argc, &argv);

  // Calculate letterbox
  LetterboxParams lb = calculateLetterbox(SOURCE_WIDTH, SOURCE_HEIGHT);
  log_info("Letterbox: %dx%d -> scale %.4f -> %dx%d + pad L=%d R=%d T=%d B=%d\n",
           SOURCE_WIDTH, SOURCE_HEIGHT, lb.scale, lb.scaledW, lb.scaledH,
           lb.padX, lb.padRight, lb.padY, lb.padBottom);

  // Build pipeline
  char *pipelineStr = buildPipeline(platform, pargs, lb, pargs.headless);
  log_info("Pipeline: %s\n\n", pipelineStr);

  // Initialize app data
  AppData app = {};
  app.className = getCocoClassNames();
  app.letterbox = lb;
  app.platform = platform;
  app.headless = pargs.headless;
  app.numFrames = pargs.numFrames;

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
  app.timing.e2ePipeline.reset();

  app.throughput.reset();
  app.ptsTracker.init();

  // Setup bus callback context
  bool startedOnce = false;
  app.busCtx.pipeline = NULL;  // Set after pipeline creation
  app.busCtx.loop = NULL;
  app.busCtx.playing = &app.playing;
  app.busCtx.startedOnce = &startedOnce;
  app.busCtx.videoLoop = !pargs.video.empty() && pargs.image.empty();
  app.busCtx.videoRate = pargs.speed;
  app.busCtx.printTiming = NULL;  // print after pipeline teardown
  app.busCtx.appData = &app;

  // Create pipeline
  app.loop = g_main_loop_new(NULL, FALSE);
  app.gstPipeline = gst_parse_launch(pipelineStr, NULL);
  g_free(pipelineStr);

  if (!app.gstPipeline) {
    log_error("Failed to create pipeline\n");
    return 1;
  }

  app.busCtx.pipeline = app.gstPipeline;
  app.busCtx.loop = app.loop;

  // Connect signals
  app.bus = gst_element_get_bus(app.gstPipeline);
  gst_bus_add_signal_watch(app.bus);
  g_signal_connect(app.bus, "message", G_CALLBACK(commonBusCallback), &app.busCtx);

  GstElement *tsink = gst_bin_get_by_name(GST_BIN(app.gstPipeline), "inferenceOutput");
  g_signal_connect(tsink, "new-data", G_CALLBACK(newDataCallback), &app);
  gst_object_unref(tsink);

  // Cairo overlay (only when displaying)
  if (!pargs.headless) {
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
  g_unix_signal_add(SIGINT, commonSigintHandler, &app.busCtx);

  // Run
  gst_element_set_state(app.gstPipeline, GST_STATE_PLAYING);
  g_main_loop_run(app.loop);

  // Cleanup — tear down pipeline first so waylandsink stats print before ours
  gst_element_set_state(app.gstPipeline, GST_STATE_NULL);
  printTimingStatistics(&app);
  if (app.tensorFilter)
    gst_object_unref(app.tensorFilter);
  gst_object_unref(app.bus);
  gst_object_unref(app.gstPipeline);
  g_main_loop_unref(app.loop);
  app.ptsTracker.destroy();

  return 0;
}
