/**
 * Copyright 2026 EdgeFirst AI (Au-Zone Technologies)
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * YOLOv8n 640x640 Reference Demo — Kinara Ara-2 NPU (Standard Pipeline)
 *
 * Reference baseline for Ara-2 NPU benchmarking. Uses the standard NXP
 * 7-element preprocessing pipeline with manual dequantization and NMS.
 * No EdgeFirst dependencies — purpose is A/B comparison against the
 * EdgeFirst-optimized yolov8n_ara2 variant.
 *
 * Supports both i.MX 8M Plus (v4l2src) and i.MX 95 (libcamerasrc) via
 * the --platform flag.
 *
 * Ara-2 YOLOv8n.dvm output tensors (split format):
 *   [0] scores — uint8  [80, 8400]  (post-sigmoid class confidences)
 *   [1] boxes  — int16  [4, 8400]   (cx, cy, w, h in model pixel space)
 *
 * PREPROCESSING PIPELINE (CPU-heavy, NXP-style elements):
 *   1. G2D Scale+Colorspace: imxvideoconvert_g2d (1920x1080 → 640x360 RGBA)
 *   2. Letterbox Padding: videobox (640x360 → 640x640 with black borders)
 *   3. Colorspace: videoconvert (RGBA → RGB strip alpha channel)
 *   4. Tensor Conversion: tensor_converter (RGB frame → tensor buffer)
 *   5. Transpose: tensor_transform (HWC → CHW layout for Ara-2)
 *   6. Tensor Shift: uint8→int16, add -128, int16→int8 (avoids int8 saturation)
 *   7. Memory merge: tensor_aggregator (consolidate buffer memory blocks)
 *
 * IMPORTANT: tensor_converter must output format=static (not flexible) to
 * prevent tensor_transform from corrupting the 128-byte flexible header.
 * tensor_aggregator merges the split memory blocks that tensor_transform
 * produces into a single contiguous buffer for the Ara-2 tensor_filter.
 *
 * INFERENCE: tensor_filter framework=ara2
 *   The sub-plugin expects preprocessed int8 CHW tensors matching the
 *   model's native format. The tensor_transform chain above provides this.
 *
 * POST-PROCESSING: Manual dequantization from split tensors + NMS
 *
 * Usage:
 *   ./yolov8n_ara2_reference -m yolov8n.dvm -p imx95
 *   ./yolov8n_ara2_reference -m yolov8n.dvm -p imx8mp -v test.mp4
 *   ./yolov8n_ara2_reference -m yolov8n.dvm -p imx95 -H -n 100
 */

#include <gst/gst.h>
#include <glib.h>
#include <glib-unix.h>
#include <cairo.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <sys/time.h>
#include <vector>

#include "logging.hpp"

#include "common/yolov8_common.hpp"


#define NUM_OUTPUTS 2  /* scores + boxes tensors */


/* ─── Platform configuration ──────────────────────────────────────── */

enum Platform { PLATFORM_IMX95, PLATFORM_IMX8MP };

struct PlatformConfig {
  const char *name;
  const char *defaultCamera;
  bool usesLibcamerasrc;
};

static const PlatformConfig platformConfigs[] = {
  [PLATFORM_IMX95] = {
    .name = "i.MX 95",
    .defaultCamera = NULL,
    .usesLibcamerasrc = true,
  },
  [PLATFORM_IMX8MP] = {
    .name = "i.MX 8M Plus",
    .defaultCamera = "/dev/video3",
    .usesLibcamerasrc = false,
  },
};


/* ─── Detection result for display ────────────────────────────────── */

struct DetectionResult {
  int classId;
  float score;
  float x, y, w, h;
};


/* ─── Application data ────────────────────────────────────────────── */

struct AppData {
  GstElement *pipeline;
  GMainLoop *loop;
  GstBus *bus;
  gboolean playing;

  bool headless;

  /* Letterbox parameters */
  LetterboxParams letterbox;

  /* Timing — preprocessing stages */
  TimingMetric g2dScale;
  TimingMetric letterboxTime;
  TimingMetric colorconv;
  TimingMetric tensorConv;
  TimingMetric transpose;
  TimingMetric tensorShift;
  TimingMetric preprocTotal;

  /* Timing — inference */
  TimingMetric inference;

  /* Timing — post-processing stages */
  TimingMetric outputMmap;
  TimingMetric dequantExtract;
  TimingMetric nms;
  TimingMetric postprocTotal;

  /* Timing — end-to-end */
  TimingMetric e2ePipeline;

  /* Timestamps for pad probe timing */
  struct timeval g2dStart;
  struct timeval g2dEnd;
  struct timeval letterboxEnd;
  struct timeval colorconvEnd;
  struct timeval tensorconvEnd;
  struct timeval transposeEnd;
  struct timeval tshiftEnd;

  /* Detection statistics */
  int totalDetections;
  int framesWithDetections;
  int preNmsDetections;
  int totalFrames;
  int maxFrames;

  GstElement *tensorFilter;

  /* Detection results for display overlay */
  std::vector<DetectionResult> results;

  /* Quantization parameters (from GstNnsTensorQuantMeta) */
  double scoresScale;
  double boxesScale;
  bool quantInitialized = false;

  /* Common infrastructure */
  BusCallbackCtx busCtx;
  ThroughputTracker throughput;
  PtsTracker ptsTracker;
};


/* ─── Preprocessing timing probes ─────────────────────────────────── */

static GstPadProbeReturn
g2dSinkProbe(GstPad *, GstPadProbeInfo *info, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->g2dStart, NULL);

  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  app->ptsTracker.recordStart(buffer, app->g2dStart);
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
g2dSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->g2dEnd, NULL);
  app->g2dScale.record(timeDiffMs(app->g2dStart, app->g2dEnd));
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
letterboxSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->letterboxEnd, NULL);
  app->letterboxTime.record(timeDiffMs(app->g2dEnd, app->letterboxEnd));
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
colorconvSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->colorconvEnd, NULL);
  app->colorconv.record(timeDiffMs(app->letterboxEnd, app->colorconvEnd));
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
tensorconvSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->tensorconvEnd, NULL);
  app->tensorConv.record(timeDiffMs(app->colorconvEnd, app->tensorconvEnd));
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
transposeSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->transposeEnd, NULL);
  app->transpose.record(timeDiffMs(app->tensorconvEnd, app->transposeEnd));
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
tshiftSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->tshiftEnd, NULL);
  app->tensorShift.record(timeDiffMs(app->transposeEnd, app->tshiftEnd));
  app->preprocTotal.record(timeDiffMs(app->g2dStart, app->tshiftEnd));
  return GST_PAD_PROBE_OK;
}


/* ─── Timing report ───────────────────────────────────────────────── */

static void printTiming(void *userData)
{
  AppData *app = (AppData *)userData;

  printf("\n");
  printf("===================================================================="
         "========\n");
  printf("  REFERENCE PIPELINE — KINARA ARA-2 NPU (Standard Preprocessing)\n");
  printf("===================================================================="
         "========\n");

  printf("\n  PREPROCESSING (1920x1080 NV12 -> 640x640 CHW INT8 tensor)\n");
  printf("  ------------------------------------------------------------------"
         "----------\n");
  printMetric("1. G2D Scale + Colorspace [FUSED - HW accelerated]",
      "1920x1080 NV12 -> 640x360 RGBA (aspect-fit)", app->g2dScale);
  printMetric("2. Letterbox Padding [videobox]",
      "640x360 -> 640x640 with black borders", app->letterboxTime);
  printMetric("3. Colorspace Conversion [videoconvert]",
      "RGBA -> RGB strip alpha channel", app->colorconv);
  printMetric("4. Tensor Conversion [tensor_converter]",
      "RGB frame -> tensor buffer", app->tensorConv);
  printMetric("5. Transpose [tensor_transform]",
      "HWC -> CHW layout for Ara-2", app->transpose);
  printMetric("6. Tensor Shift [typecast + add]",
      "uint8->int16-128->int8 (two-step to avoid saturation)", app->tensorShift);

  if (app->preprocTotal.count > 0) {
    printf("\n  >> PREPROCESSING TOTAL\n");
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms\n",
        app->preprocTotal.avg(),
        app->preprocTotal.minMs,
        app->preprocTotal.maxMs);
  }

  printf("\n===================================================================="
         "========\n");
  printf("\n  INFERENCE (Kinara Ara-2 NPU)\n");
  printf("  ------------------------------------------------------------------"
         "----------\n");
  if (app->inference.count > 0) {
    printf("     tensor_filter framework=ara2 (native int8 CHW)\n");
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms"
           "  [%d samples]\n",
        app->inference.avg(), app->inference.minMs,
        app->inference.maxMs, app->inference.count);
  }

  printf("\n===================================================================="
         "========\n");
  printf("\n  POST-PROCESSING (split tensors -> detection boxes)\n");
  printf("  ------------------------------------------------------------------"
         "----------\n");
  printMetric("1. Output Tensor Read [gst_memory_map]",
      "mmap 2 output tensors (scores + boxes)", app->outputMmap);
  printMetric("2. Dequantization + Box Extraction",
      "uint8 scores + int16 boxes, 8400 candidates",
      app->dequantExtract);

  if (app->nms.count > 0) {
    printf("  3. Non-Maximum Suppression [class-agnostic NMS]\n");
    printf("     (IoU threshold: %.2f, confidence threshold: %.2f)\n",
        NMS_IOU_THRESHOLD, CONF_THRESHOLD);
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms"
           "  [%d frames]\n",
        app->nms.avg(), app->nms.minMs,
        app->nms.maxMs, app->nms.count);
  }

  if (app->postprocTotal.count > 0) {
    printf("\n  >> POST-PROCESSING TOTAL\n");
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms\n",
        app->postprocTotal.avg(),
        app->postprocTotal.minMs,
        app->postprocTotal.maxMs);
  }

  printf("\n===================================================================="
         "========\n");
  printf("\n  END-TO-END TIMING ANALYSIS\n");
  printf("  ------------------------------------------------------------------"
         "----------\n");

  double sumPreproc = app->preprocTotal.avg();
  double sumInference = app->inference.avg();
  double sumPostproc = app->postprocTotal.avg();
  double sumNN = sumPreproc + sumInference + sumPostproc;

  if (app->e2ePipeline.count > 0) {
    printf("\n  1. E2E NN Pipeline (PTS-correlated: g2d sink -> "
           "post-processing)\n");
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms"
           "  [%d frames]\n",
        app->e2ePipeline.avg(), app->e2ePipeline.minMs,
        app->e2ePipeline.maxMs, app->e2ePipeline.count);
  }

  printf("\n  2. Sum of Stages\n");
  printf("     +-----------------------------------------------------------"
         "----+\n");
  printf("     | Preprocessing (G2D->letterbox->shift):      %7.3f ms"
         "          |\n",
      sumPreproc);
  printf("     | Inference (Ara-2 NPU):                     %7.3f ms"
         "          |\n",
      sumInference);
  printf("     | Post-processing (mmap->NMS):              %7.3f ms"
         "          |\n",
      sumPostproc);
  printf("     +-----------------------------------------------------------"
         "----+\n");
  printf("     | NN Branch Total (preproc+inf+postproc):   %7.3f ms"
         "          |\n",
      sumNN);
  printf("     +-----------------------------------------------------------"
         "----+\n");

  if (app->e2ePipeline.count > 0) {
    double e2eAvg = app->e2ePipeline.avg();
    double coverage = (sumNN / e2eAvg) * 100.0;
    printf("\n  3. Timing Validation\n");
    printf("     E2E NN Pipeline: %7.3f ms  |  Sum of NN Stages: "
           "%7.3f ms\n",
        e2eAvg, sumNN);
    printf("     Coverage: %6.1f%%\n", coverage);
  }

  if (app->throughput.metric.count > 0) {
    double avgMs = app->throughput.metric.avg();
    printf("\n  4. Frame Throughput\n");
    printf("     Average: %7.3f ms (%5.1f FPS)  |  Min: %7.3f ms  |"
           "  Max: %7.3f ms\n",
        avgMs, 1000.0 / avgMs, app->throughput.metric.minMs,
        app->throughput.metric.maxMs);
    printf("     Frames: %d\n", app->throughput.metric.count);
  }

  printf("\n===================================================================="
         "========\n");
  printf("\n  DETECTION STATISTICS\n");
  printf("  ------------------------------------------------------------------"
         "----------\n");
  printf("     Pre-NMS candidates:      %6d  (avg %.1f/frame)\n",
      app->preNmsDetections,
      app->totalFrames > 0
          ? (double)app->preNmsDetections / app->totalFrames
          : 0);
  printf("     Post-NMS detections:     %6d  (avg %.1f/frame)\n",
      app->totalDetections,
      app->framesWithDetections > 0
          ? (double)app->totalDetections / app->framesWithDetections
          : 0);
  printf("     Frames with detections:  %6d / %d\n",
      app->framesWithDetections, app->totalFrames);
  printf("\n===================================================================="
         "========\n");
}


/* ─── Cairo draw callback ─────────────────────────────────────────── */

static void drawCallback(GstElement *, cairo_t *cr, guint64, guint64, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  if (app->results.empty()) return;

  cairo_set_source_rgb(cr, 0, 1, 0);
  cairo_set_line_width(cr, 2.0);

  for (auto &det : app->results) {
    cairo_rectangle(cr, det.x, det.y, det.w, det.h);
    cairo_move_to(cr, det.x + 5, det.y + 15);
    const char *name = (det.classId >= 0 && det.classId < NUM_CLASSES)
        ? cocoClassNames[det.classId] : "?";
    char label[64];
    snprintf(label, sizeof(label), "%s %.0f%%", name, det.score * 100);
    cairo_show_text(cr, label);
  }
  cairo_stroke(cr);
}


/* ─── tensor_sink new-data callback ───────────────────────────────── */

static void
newDataCallback(GstElement *, GstBuffer *buffer, gpointer user_data)
{
  struct timeval callbackStart, mmapEnd, dequantEnd, nmsEnd;
  gettimeofday(&callbackStart, NULL);

  AppData *app = (AppData *)user_data;

  /* Frame-to-frame interval */
  app->throughput.tick(callbackStart);

  /* Inference latency from tensor_filter */
  queryInferenceLatency(app->tensorFilter, app->inference);

  /* Validate buffer — expect 2 memory blocks (scores + boxes) */
  if (!GST_IS_BUFFER(buffer)) {
    log_error("Invalid buffer\n");
    return;
  }
  guint n_mem = gst_buffer_n_memory(buffer);
  if (n_mem != NUM_OUTPUTS) {
    log_error("Expected %d tensors, got %u\n", NUM_OUTPUTS, n_mem);
    return;
  }

  /* Map both output tensors */
  GstMapInfo score_map, box_map;
  GstMemory *score_mem = gst_buffer_peek_memory(buffer, 0);
  GstMemory *box_mem = gst_buffer_peek_memory(buffer, 1);

  if (!gst_memory_map(score_mem, &score_map, GST_MAP_READ)) {
    log_error("Cannot map score tensor\n");
    return;
  }
  if (!gst_memory_map(box_mem, &box_map, GST_MAP_READ)) {
    log_error("Cannot map box tensor\n");
    gst_memory_unmap(score_mem, &score_map);
    return;
  }

  gettimeofday(&mmapEnd, NULL);
  app->outputMmap.record(timeDiffMs(callbackStart, mmapEnd));

  /* Extract quantization parameters from GstNnsTensorQuantMeta on first frame */
  if (!app->quantInitialized) {
    QuantParams scores_qp, boxes_qp;
    if (!extractQuantParams(buffer, 0, scores_qp) ||
        !extractQuantParams(buffer, 1, boxes_qp)) {
      log_error("No quant meta — requires updated NNStreamer\n");
      gst_memory_unmap(score_mem, &score_map);
      gst_memory_unmap(box_mem, &box_map);
      return;
    }
    app->scoresScale = scores_qp.scale;
    app->boxesScale = boxes_qp.scale;
    app->quantInitialized = true;

    log_info("Scores quant: scale=%g zero_point=%" G_GINT64_FORMAT "\n",
             scores_qp.scale, scores_qp.zeroPoint);
    log_info("Boxes  quant: scale=%g zero_point=%" G_GINT64_FORMAT "\n",
             boxes_qp.scale, boxes_qp.zeroPoint);
  }

  const uint8_t *scores = (const uint8_t *)score_map.data;
  const int16_t *boxes = (const int16_t *)box_map.data;

  /* Dequantization + box extraction from split tensors. */
  struct Detection {
    float x, y, w, h;
    float conf;
    int classId;
  };

  std::vector<Detection> candidates;

  for (int bIdx = 0; bIdx < NUM_TOTAL_BOXES; bIdx++) {
    int maxClassIdx = -1;
    uint8_t maxScore = 0;

    for (int cIdx = 0; cIdx < NUM_CLASSES; cIdx++) {
      uint8_t s = scores[cIdx * NUM_TOTAL_BOXES + bIdx];
      if (s > maxScore) {
        maxScore = s;
        maxClassIdx = cIdx;
      }
    }

    float conf = maxScore * app->scoresScale;
    if (conf < CONF_THRESHOLD)
      continue;

    float cx = boxes[0 * NUM_TOTAL_BOXES + bIdx] * app->boxesScale;
    float cy = boxes[1 * NUM_TOTAL_BOXES + bIdx] * app->boxesScale;
    float bw = boxes[2 * NUM_TOTAL_BOXES + bIdx] * app->boxesScale;
    float bh = boxes[3 * NUM_TOTAL_BOXES + bIdx] * app->boxesScale;

    /* Letterbox reverse mapping: model pixel coords -> source coords */
    float px = (cx - app->letterbox.padX) / app->letterbox.scale;
    float py = (cy - app->letterbox.padY) / app->letterbox.scale;
    float pw = bw / app->letterbox.scale;
    float ph = bh / app->letterbox.scale;

    Detection det;
    det.x = px - pw / 2.0f;
    det.y = py - ph / 2.0f;
    det.w = pw;
    det.h = ph;
    det.conf = conf;
    det.classId = maxClassIdx;
    candidates.push_back(det);
  }

  /* Unmap tensors AFTER all reads complete (DMA-BUF unmap pitfall) */
  gst_memory_unmap(score_mem, &score_map);
  gst_memory_unmap(box_mem, &box_map);

  gettimeofday(&dequantEnd, NULL);
  app->dequantExtract.record(timeDiffMs(mmapEnd, dequantEnd));
  app->preNmsDetections += candidates.size();

  /* Class-agnostic NMS (matches EdgeFirst HAL behavior for comparison) */
  std::sort(candidates.begin(), candidates.end(),
      [](const Detection &a, const Detection &b) {
        return a.conf > b.conf;
      });

  std::vector<Detection> nmsResults;
  std::vector<bool> suppressed(candidates.size(), false);

  for (size_t i = 0; i < candidates.size(); i++) {
    if (suppressed[i])
      continue;
    nmsResults.push_back(candidates[i]);

    for (size_t j = i + 1; j < candidates.size(); j++) {
      if (suppressed[j])
        continue;

      float x1 = std::max(candidates[i].x, candidates[j].x);
      float y1 = std::max(candidates[i].y, candidates[j].y);
      float x2 = std::min(candidates[i].x + candidates[i].w,
          candidates[j].x + candidates[j].w);
      float y2 = std::min(candidates[i].y + candidates[i].h,
          candidates[j].y + candidates[j].h);
      float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
      float area_i = candidates[i].w * candidates[i].h;
      float area_j = candidates[j].w * candidates[j].h;
      float iou_val = inter / (area_i + area_j - inter + 1e-6f);

      if (iou_val > NMS_IOU_THRESHOLD)
        suppressed[j] = true;
    }
  }

  gettimeofday(&nmsEnd, NULL);
  app->nms.record(timeDiffMs(dequantEnd, nmsEnd));

  app->totalFrames++;

  // Store detections for display overlay
  app->results.clear();
  for (auto &det : nmsResults) {
    app->results.push_back({det.classId, det.conf, det.x, det.y, det.w, det.h});
  }

  // Frame limit — stop after N frames and print timing
  if (app->maxFrames > 0 && app->totalFrames >= app->maxFrames) {
    if (!nmsResults.empty()) {
      app->framesWithDetections++;
      app->totalDetections += nmsResults.size();
    }
    app->postprocTotal.record(timeDiffMs(callbackStart, nmsEnd));
    g_main_loop_quit(app->loop);
    return;
  }

  if (!nmsResults.empty()) {
    app->framesWithDetections++;
    app->totalDetections += nmsResults.size();

    /* Print first frame detections */
    static bool first_print = true;
    if (first_print) {
      log_info("Detected objects (%zu):\n", nmsResults.size());
      for (size_t d = 0; d < nmsResults.size(); d++) {
        const char *name = (nmsResults[d].classId < NUM_CLASSES)
            ? cocoClassNames[nmsResults[d].classId]
            : "unknown";
        log_info("  - %s (%.2f) at (%.0f,%.0f) %.0fx%.0f\n",
            name, nmsResults[d].conf,
            nmsResults[d].x, nmsResults[d].y,
            nmsResults[d].w, nmsResults[d].h);
      }
      log_info("\n");
      first_print = false;
    }
  }

  app->postprocTotal.record(timeDiffMs(callbackStart, nmsEnd));

  /* PTS-correlated E2E */
  struct timeval ptsStart;
  if (app->ptsTracker.consumeStart(buffer, ptsStart)) {
    app->e2ePipeline.record(timeDiffMs(ptsStart, nmsEnd));
  }
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
    {"g2d_scale", "src", g2dSrcProbe},
    {"letterbox", "src", letterboxSrcProbe},
    {"colorconv", "src", colorconvSrcProbe},
    {"tconv", "src", tensorconvSrcProbe},
    {"transpose", "src", transposeSrcProbe},
    {"tagg", "src", tshiftSrcProbe},
  };

  for (auto &p : probes) {
    GstElement *elem = gst_bin_get_by_name(GST_BIN(pipeline), p.elementName);
    if (elem) {
      GstPad *pad = gst_element_get_static_pad(elem, p.padName);
      if (pad) {
        gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER,
            p.callback, app, NULL);
        gst_object_unref(pad);
      }
      gst_object_unref(elem);
    }
  }
}


/* ─── Pipeline construction ───────────────────────────────────────── */

static char *buildPipeline(Platform platform, const ParsedArgs &pargs,
                           const LetterboxParams &lb, bool headless)
{
  const PlatformConfig &plat = platformConfigs[platform];

  // Source element
  InputSource srcType = determineInputSource(pargs, plat.usesLibcamerasrc);
  char *srcStr = buildSourceElement(srcType, pargs);

  // NN branch: G2D aspect-fit resize + videobox letterbox padding + preprocessing + inference + sink
  char *pipeline;
  if (headless) {
    pipeline = g_strdup_printf(
        "%s ! queue name=thread-nn leaky=2 max-size-buffers=2 ! "
        "imxvideoconvert_g2d name=g2d_scale ! "
        "video/x-raw,width=%d,height=%d,format=RGBA ! "
        "videobox name=letterbox left=%d right=%d top=%d bottom=%d fill=black ! "
        "videoconvert name=colorconv ! video/x-raw,format=RGB ! "
        "tensor_converter name=tconv ! "
        "other/tensor,format=static,dimension=3:%d:%d:1,type=uint8 ! "
        "tensor_transform name=transpose mode=transpose option=1:2:0:3 ! "
        "tensor_transform name=tshift1 mode=arithmetic "
        "option=typecast:int16,add:-128 ! "
        "tensor_transform name=tshift2 mode=typecast option=int8 ! "
        "tensor_aggregator name=tagg frames-in=1 frames-out=1 frames-flush=1 ! "
        "tensor_filter name=tfilter framework=ara2 model=%s "
        "custom=EnableStats:true latency=1 ! "
        "tensor_sink name=inferenceOutput",
        srcStr,
        lb.scaledW, lb.scaledH,
        -lb.padX, -lb.padRight, -lb.padY, -lb.padBottom,
        MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
        pargs.model.c_str());
  } else {
    const char *syncMode = (!pargs.video.empty() || !pargs.image.empty()) ? "true" : "false";
    pipeline = g_strdup_printf(
        "%s ! tee name=t "
        "t. ! queue name=thread-nn leaky=2 max-size-buffers=2 ! "
        "imxvideoconvert_g2d name=g2d_scale ! "
        "video/x-raw,width=%d,height=%d,format=RGBA ! "
        "videobox name=letterbox left=%d right=%d top=%d bottom=%d fill=black ! "
        "videoconvert name=colorconv ! video/x-raw,format=RGB ! "
        "tensor_converter name=tconv ! "
        "other/tensor,format=static,dimension=3:%d:%d:1,type=uint8 ! "
        "tensor_transform name=transpose mode=transpose option=1:2:0:3 ! "
        "tensor_transform name=tshift1 mode=arithmetic "
        "option=typecast:int16,add:-128 ! "
        "tensor_transform name=tshift2 mode=typecast option=int8 ! "
        "tensor_aggregator name=tagg frames-in=1 frames-out=1 frames-flush=1 ! "
        "tensor_filter name=tfilter framework=ara2 model=%s "
        "custom=EnableStats:true latency=1 ! "
        "tensor_sink name=inferenceOutput "
        "t. ! queue name=thread-img max-size-buffers=2 ! "
        "imxvideoconvert_g2d ! "
        "cairooverlay name=cairo ! "
        "waylandsink sync=%s",
        srcStr,
        lb.scaledW, lb.scaledH,
        -lb.padX, -lb.padRight, -lb.padY, -lb.padBottom,
        MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
        pargs.model.c_str(),
        syncMode);
  }
  g_free(srcStr);

  return pipeline;
}


/* ─── Main ────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
  ParsedArgs pargs;
  pargs.camera = "";  // Will be set from platform default

  uint32_t flags = ARG_MODEL | ARG_CAMERA | ARG_VIDEO | ARG_IMAGE |
                   ARG_HEADLESS | ARG_INSTRUMENTED | ARG_NUM_FRAMES | ARG_PLATFORM;

  int ret = parseArgs(argc, argv, flags,
      "YOLOv8n 640x640 Reference Pipeline for Kinara Ara-2 NPU", pargs);
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
    setupImx95Environment();

  gst_init(&argc, &argv);

  log_info("YOLOv8n 640x640 Reference Pipeline — Ara-2 NPU on %s\n", plat.name);
  log_info("Model: %s\n", pargs.model.c_str());
  if (!pargs.image.empty()) {
    log_info("Input: image (%s)\n", pargs.image.c_str());
  } else if (!pargs.video.empty()) {
    log_info("Input: video (%s)\n", pargs.video.c_str());
  } else {
    log_info("Input: camera (%s)\n", pargs.camera.empty() ? "libcamerasrc" : pargs.camera.c_str());
  }
  if (pargs.numFrames > 0)
    log_info("Frames: %d\n", pargs.numFrames);
  log_info("Mode: %s\n", pargs.headless ? "headless" : "display");

  // Calculate letterbox
  LetterboxParams lb = calculateLetterbox(SOURCE_WIDTH, SOURCE_HEIGHT);
  log_info("Letterbox: %dx%d -> scale %.4f -> %dx%d + pad L=%d R=%d T=%d B=%d\n",
           SOURCE_WIDTH, SOURCE_HEIGHT, lb.scale, lb.scaledW, lb.scaledH,
           lb.padX, lb.padRight, lb.padY, lb.padBottom);

  /* Build pipeline */
  char *pipelineStr = buildPipeline(platform, pargs, lb, pargs.headless);
  log_info("Pipeline: %s\n\n", pipelineStr);

  AppData app = {};
  app.letterbox = lb;
  app.headless = pargs.headless;
  app.maxFrames = pargs.numFrames;
  app.g2dScale.reset();
  app.letterboxTime.reset();
  app.colorconv.reset();
  app.tensorConv.reset();
  app.transpose.reset();
  app.tensorShift.reset();
  app.preprocTotal.reset();
  app.inference.reset();
  app.outputMmap.reset();
  app.dequantExtract.reset();
  app.nms.reset();
  app.postprocTotal.reset();
  app.e2ePipeline.reset();
  app.throughput.reset();
  app.ptsTracker.init();

  // Setup bus callback context
  bool startedOnce = false;
  app.busCtx.playing = &app.playing;
  app.busCtx.startedOnce = pargs.headless ? NULL : &startedOnce;
  app.busCtx.videoLoop = !pargs.video.empty() && pargs.image.empty();
  app.busCtx.videoRate = 1.0;
  app.busCtx.printTiming = NULL;  // print after pipeline teardown
  app.busCtx.appData = &app;

  app.loop = g_main_loop_new(NULL, FALSE);
  app.pipeline = gst_parse_launch(pipelineStr, NULL);
  g_free(pipelineStr);

  if (!app.pipeline) {
    log_error("Failed to create pipeline\n");
    return 1;
  }

  app.busCtx.pipeline = app.pipeline;
  app.busCtx.loop = app.loop;

  /* Bus */
  app.bus = gst_element_get_bus(app.pipeline);
  gst_bus_add_signal_watch(app.bus);
  g_signal_connect(app.bus, "message", G_CALLBACK(commonBusCallback), &app.busCtx);

  /* tensor_sink */
  GstElement *tsink = gst_bin_get_by_name(GST_BIN(app.pipeline),
      "inferenceOutput");
  g_signal_connect(tsink, "new-data", G_CALLBACK(newDataCallback), &app);
  gst_object_unref(tsink);

  /* Cairo overlay (only when displaying) */
  if (!pargs.headless) {
    GstElement *cairo = gst_bin_get_by_name(GST_BIN(app.pipeline), "cairo");
    if (cairo) {
      g_signal_connect(cairo, "draw", G_CALLBACK(drawCallback), &app);
      gst_object_unref(cairo);
    }
  }

  /* tensor_filter for latency */
  app.tensorFilter = gst_bin_get_by_name(GST_BIN(app.pipeline), "tfilter");

  /* Install timing probes */
  installProbes(app.pipeline, &app);

  g_unix_signal_add(SIGINT, commonSigintHandler, &app.busCtx);

  /* Run */
  gst_element_set_state(app.pipeline, GST_STATE_PLAYING);
  g_main_loop_run(app.loop);

  /* Cleanup — tear down pipeline first so waylandsink stats print before ours */
  gst_element_set_state(app.pipeline, GST_STATE_NULL);
  printTiming(&app);
  if (app.tensorFilter)
    gst_object_unref(app.tensorFilter);
  gst_object_unref(app.bus);
  gst_object_unref(app.pipeline);
  g_main_loop_unref(app.loop);
  app.ptsTracker.destroy();

  return 0;
}
