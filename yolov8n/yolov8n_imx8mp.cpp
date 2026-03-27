/**
 * Copyright 2025 NXP
 * Copyright 2026 EdgeFirst AI (Au-Zone Technologies)
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * YOLOv8n 640x640 Camera Demo for i.MX 8M Plus — Full EdgeFirst Optimized
 *
 * EDGEFIRST OPTIMIZATIONS vs reference pipeline:
 *   - Framework: tensorflow2-lite (V2 invoke path with DMA-BUF support)
 *   - CameraAdaptor:rgba — NPU handles RGBA→RGB slice + UINT8→INT8 quantization
 *   - G2D keep-ratio=true — fused scale+colorspace+letterbox in single HW blit
 *   - DMA-BUF zero-copy: G2D writes directly to delegate-owned buffer
 *   - EdgeFirst HAL for quantized NMS (no CPU float dequantization)
 *   - Eliminated: videobox, videoconvert, tensor_transform #1 & #2
 *
 * PREPROCESSING (2 elements vs 6 in reference):
 *   1. G2D Scale+Colorspace+Letterbox: imxvideoconvert_g2d keep-ratio=true
 *      (1920x1080 NV12 → 640x640 RGBA, writes to DMA-BUF)
 *   2. Tensor Conversion: tensor_converter (RGBA frame → tensor buffer)
 *
 * INFERENCE: VSI NPU via tensorflow2-lite + libvx_delegate.so + CameraAdaptor
 *   - NPU Slice op extracts RGB from RGBA
 *   - NPU DataConvert handles UINT8→INT8
 *
 * POST-PROCESSING: EdgeFirst HAL decoder (quantized NMS, no dequantization)
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
#include <unistd.h>
#include <vector>

#include "logging.hpp"

#include "common/yolov8_common.hpp"

#include <edgefirst/hal.h>

#define DEFAULT_CAMERA_DEVICE "/dev/video3"


/* ─── Segmentation Context (HAL-dependent, imx8mp only) ──────────── */

static void initSegContext(SegContext &ctx, int width, int height) {
  ctx.processor = hal_image_processor_new();
  if (!ctx.processor) {
    g_printerr("Failed to create HAL image processor\n");
    return;
  }
  ctx.overlay = hal_image_processor_create_image(ctx.processor,
                                                  width, height,
                                                  HAL_PIXEL_FORMAT_RGBA,
                                                  HAL_DTYPE_U8);
  if (!ctx.overlay) {
    g_printerr("Failed to create PBO overlay image %dx%d\n", width, height);
    hal_image_processor_free(ctx.processor);
    ctx.processor = NULL;
    return;
  }
  g_mutex_init(&ctx.mutex);
  ctx.overlayReady = false;
  ctx.segMode = false;
  ctx.overlayWidth = width;
  ctx.overlayHeight = height;
}

static void freeSegContext(SegContext &ctx) {
  if (ctx.overlay) {
    hal_tensor_free(ctx.overlay);
    ctx.overlay = NULL;
  }
  if (ctx.processor) {
    hal_image_processor_free(ctx.processor);
    ctx.processor = NULL;
  }
  g_mutex_clear(&ctx.mutex);
}

// HAL JSON template — quantization filled from GstNnsTensorQuantMeta at runtime
// Detection-only model: single [1, 84, 8400] output
static const char *HAL_DET_CONFIG_FMT =
    "{\"outputs\":[{\"decoder\":\"ultralytics\","
    "\"type\":\"detection\","
    "\"shape\":[1,84,8400],"
    "\"quantization\":[%g,%" G_GINT64_FORMAT "],"
    "\"dshape\":[[\"batch\",1],[\"num_features\",84],[\"num_boxes\",8400]]}],"
    "\"nms\":\"class_agnostic\"}";

// Segmentation model: 4 split outputs (scores, protos, boxes, mask_coefficients)
// NNStreamer output order for yolov8n-seg split model:
//   mem[0]: scores   [1, 80, 8400]      INT8
//   mem[1]: protos   [1, 160, 160, 32]  INT8
//   mem[2]: boxes    [1, 4, 8400]       INT8
//   mem[3]: coeffs   [1, 32, 8400]      INT8
static const char *HAL_SEG_CONFIG_FMT =
    "{\"outputs\":["
    "{\"decoder\":\"ultralytics\",\"type\":\"scores\","
    "\"shape\":[1,80,8400],"
    "\"dshape\":[[\"batch\",1],[\"num_classes\",80],[\"num_boxes\",8400]],"
    "\"quantization\":[%g,%" G_GINT64_FORMAT "]},"
    "{\"decoder\":\"ultralytics\",\"type\":\"protos\","
    "\"shape\":[1,160,160,32],"
    "\"dshape\":[[\"batch\",1],[\"height\",160],[\"width\",160],[\"num_protos\",32]],"
    "\"quantization\":[%g,%" G_GINT64_FORMAT "]},"
    "{\"decoder\":\"ultralytics\",\"type\":\"boxes\","
    "\"shape\":[1,4,8400],"
    "\"dshape\":[[\"batch\",1],[\"box_coords\",4],[\"num_boxes\",8400]],"
    "\"quantization\":[%g,%" G_GINT64_FORMAT "]},"
    "{\"decoder\":\"ultralytics\",\"type\":\"mask_coefficients\","
    "\"shape\":[1,32,8400],"
    "\"dshape\":[[\"batch\",1],[\"num_protos\",32],[\"num_boxes\",8400]],"
    "\"quantization\":[%g,%" G_GINT64_FORMAT "]}"
    "],\"nms\":\"class_agnostic\"}";


/* ─── Timing statistics ───────────────────────────────────────────── */

typedef struct {
  // Preprocessing (EdgeFirst: only 2 stages)
  TimingMetric g2dScale;       // G2D resize + NV12→RGBA + letterbox (FUSED, DMA-BUF)
  TimingMetric tensorConv;     // tensor_converter (RGBA → tensor)
  TimingMetric preprocTotal;   // Total preprocessing

  // Inference (includes CameraAdaptor Slice + DataConvert on NPU)
  TimingMetric inference;

  // Post-processing (HAL decoder)
  TimingMetric halDecode;      // HAL decoder (quantized NMS) — det mode
  TimingMetric halDrawMasks;   // HAL decoder draw_masks (fused decode+render) — seg mode
  TimingMetric postprocTotal;

  // Rendering
  TimingMetric cairoDraw;

  // End-to-end (PTS-correlated)
  TimingMetric e2ePipeline;

  // Timestamps
  struct timeval g2dStart;
  struct timeval g2dEnd;
  struct timeval tensorconvEnd;

  // Detection statistics
  int totalDetections;
  int framesWithDetections;
} TimingStats;


/* ─── Application data ────────────────────────────────────────────── */

struct DetectionResult {
  int classId;
  float score;
  float x, y, w, h;
};

typedef struct {
  GstElement *gstPipeline;
  GMainLoop *loop;
  GstBus *bus;
  gboolean playing;
  std::vector<DetectionResult> results;
  TimingStats timing;
  GstElement *tensorFilter;

  LetterboxParams letterbox;
  bool headless;
  bool instrumented;

  // Common infrastructure
  BusCallbackCtx busCtx;
  ThroughputTracker throughput;
  PtsTracker ptsTracker;

  // HAL decoder
  hal_decoder *decoder;
  bool decoderInitFailed;

  // Segmentation overlay
  SegContext seg;

  // Frame counting
  int numFrames;
  int frameCount;
  bool timingPrinted;
} AppData;


/* ─── Pad probes ──────────────────────────────────────────────────── */

static GstPadProbeReturn g2dSinkProbe(GstPad *, GstPadProbeInfo *info, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->timing.g2dStart, NULL);

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

static GstPadProbeReturn tensorconvSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->timing.tensorconvEnd, NULL);
  app->timing.tensorConv.record(timeDiffMs(app->timing.g2dEnd, app->timing.tensorconvEnd));
  app->timing.preprocTotal.record(timeDiffMs(app->timing.g2dStart, app->timing.tensorconvEnd));
  return GST_PAD_PROBE_OK;
}


/* ─── Timing report ───────────────────────────────────────────────── */

static void printTimingStatistics(void *userData)
{
  AppData *app = (AppData *)userData;
  if (app->timingPrinted) return;
  app->timingPrinted = true;

  printf("\n");
  printf("==============================================================================\n");
  if (app->headless)
    printf("  EDGEFIRST OPTIMIZED — i.MX 8M Plus — HEADLESS\n");
  else
    printf("  EDGEFIRST OPTIMIZED — i.MX 8M Plus — FULL PIPELINE\n");
  printf("==============================================================================\n");

  if (app->instrumented) {
    printf("\n  PREPROCESSING (1920x1080 NV12 -> 640x640 RGBA tensor)\n");
    printf("  NOTE: G2D keep-ratio=true handles letterbox (clear+blit, one-time fill)\n");
    printf("  NOTE: CameraAdaptor handles RGBA->RGB + UINT8->INT8 on NPU\n");
    printf("  --------------------------------------------------------------------------\n");
    printMetric("1. G2D Scale + Colorspace + Letterbox [FUSED - HW, DMA-BUF]",
                "1920x1080 NV12 -> 640x640 RGBA with keep-ratio=true",
                app->timing.g2dScale);
    printMetric("2. Tensor Conversion [tensor_converter]",
                "RGBA frame -> tensor buffer", app->timing.tensorConv);

    printf("\n  --- ELIMINATED (moved to HW / NPU) ---\n");
    printf("  [x] videobox (CPU letterbox)         -- G2D keep-ratio=true\n");
    printf("  [x] videoconvert (RGBA->RGB)         -- NPU Slice (CameraAdaptor)\n");
    printf("  [x] tensor_transform #1 & #2         -- NPU DataConvert\n");

    if (app->timing.preprocTotal.count > 0) {
      printf("\n  >> PREPROCESSING TOTAL\n");
      printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms\n",
             app->timing.preprocTotal.avg(),
             app->timing.preprocTotal.minMs,
             app->timing.preprocTotal.maxMs);
    }
  }

  printf("\n==============================================================================\n");
  printf("\n  INFERENCE (VSI NPU + CameraAdaptor)\n");
  printf("  --------------------------------------------------------------------------\n");
  if (app->timing.inference.count > 0) {
    printf("     tensorflow2-lite + libvx_delegate.so + CameraAdaptor:rgba\n");
    printf("     (includes NPU Slice RGBA->RGB + DataConvert UINT8->INT8)\n");
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d samples]\n",
           app->timing.inference.avg(), app->timing.inference.minMs,
           app->timing.inference.maxMs, app->timing.inference.count);
  }

  printf("\n==============================================================================\n");
  if (app->seg.segMode)
    printf("\n  POST-PROCESSING (EdgeFirst HAL — segmentation + quantized NMS)\n");
  else
    printf("\n  POST-PROCESSING (EdgeFirst HAL — quantized NMS, no dequantization)\n");
  printf("  --------------------------------------------------------------------------\n");
  printMetric("HAL decoder (dequant + NMS in one call)",
              "operates on INT8 tensors directly", app->timing.halDecode);
  printMetric("HAL draw_masks (fused decode + GPU mask render)",
              "PBO-accelerated segmentation overlay", app->timing.halDrawMasks);
  if (app->timing.postprocTotal.count > 0) {
    printf("\n  >> POST-PROCESSING TOTAL\n");
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms\n",
           app->timing.postprocTotal.avg(),
           app->timing.postprocTotal.minMs,
           app->timing.postprocTotal.maxMs);
  }

  printf("\n==============================================================================\n");
  printf("\n  END-TO-END\n");
  printf("  --------------------------------------------------------------------------\n");

  double sumPreproc = app->timing.preprocTotal.avg();
  double sumInference = app->timing.inference.avg();
  double sumPostproc = app->timing.postprocTotal.avg();
  double sumNN = sumPreproc + sumInference + sumPostproc;

  if (app->timing.e2ePipeline.count > 0) {
    printf("\n  E2E NN Pipeline (PTS-correlated: g2d sink -> post-processing)\n");
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d frames]\n",
           app->timing.e2ePipeline.avg(), app->timing.e2ePipeline.minMs,
           app->timing.e2ePipeline.maxMs, app->timing.e2ePipeline.count);

    double coverage = (sumNN / app->timing.e2ePipeline.avg()) * 100.0;
    printf("     Sum of stages: %7.3f ms  |  Coverage: %.1f%%\n", sumNN, coverage);
  }

  if (app->throughput.metric.count > 0) {
    double avgMs = app->throughput.metric.avg();
    printf("\n  Frame Throughput:\n");
    printf("     Average: %7.3f ms (%5.1f FPS)  |  Frames: %d\n",
           avgMs, 1000.0 / avgMs, app->throughput.metric.count);
  }

  printf("\n  Detection Statistics:\n");
  printf("     Post-NMS detections:     %6d  (avg %.1f/frame)\n",
         app->timing.totalDetections,
         app->timing.framesWithDetections > 0
             ? (double)app->timing.totalDetections / app->timing.framesWithDetections : 0);
  printf("     Frames with detections:  %6d / %d\n",
         app->timing.framesWithDetections, app->timing.postprocTotal.count);

  printf("\n==============================================================================\n");
  printf("\n  EDGEFIRST PIPELINE ARCHITECTURE\n");
  printf("  --------------------------------------------------------------------------\n");
  if (app->headless) {
    printf("  camera -> queue -> G2D(keep-ratio) -> tconv -> tensor_filter(V2) -> tensor_sink\n");
  } else {
    printf("  camera -> tee -+-> queue -> G2D(keep-ratio) -> tconv -> tensor_filter(V2)\n");
    printf("                 +-> queue -> G2D -> fakesink\n");
  }
  printf("\n  Optimizations:\n");
  printf("  * tensorflow2-lite: V2 invoke with DMA-BUF zero-copy\n");
  printf("  * G2D keep-ratio=true: HW letterbox to delegate DMA-BUF\n");
  printf("  * CameraAdaptor:rgba: NPU handles RGBA->RGB + UINT8->INT8\n");
  printf("  * EdgeFirst HAL: quantized NMS (no CPU dequantization)\n");
  if (app->seg.segMode)
    printf("  * EdgeFirst HAL: GPU-accelerated segmentation mask overlay (PBO)\n");
  printf("  * Eliminated: videobox, videoconvert, tensor_transform #1 & #2\n");
  printf("\n==============================================================================\n");
}


/* ─── GStreamer callbacks ─────────────────────────────────────────── */

static void newDataCallback(GstElement *, GstBuffer *buffer, gpointer user_data)
{
  struct timeval callbackStart, halStart, halEnd;
  gettimeofday(&callbackStart, NULL);

  AppData *app = (AppData *)user_data;

  // Frame-to-frame interval
  app->throughput.tick(callbackStart);

  // Inference latency
  queryInferenceLatency(app->tensorFilter, app->timing.inference);

  // Validate buffer
  if (!GST_IS_BUFFER(buffer)) { log_error("Invalid buffer\n"); return; }
  guint n_mem = gst_buffer_n_memory(buffer);

  // Accept 1 (det) or 4 (seg) output tensors
  if (n_mem != 1 && n_mem != MODEL_SEG_NUM_OUTPUTS) {
    log_error("Expected 1 or %d tensors, got %u\n", MODEL_SEG_NUM_OUTPUTS, n_mem);
    return;
  }

  // Auto-detect seg mode on first buffer
  bool isSeg = (n_mem == MODEL_SEG_NUM_OUTPUTS && app->seg.processor != NULL);
  if (!app->seg.segMode && isSeg) {
    app->seg.segMode = true;
    log_info("Segmentation model detected (%u output tensors)\n", n_mem);
  }

  // Lazy HAL decoder init — extract quant params from GstNnsTensorQuantMeta
  if (!app->decoder && !app->decoderInitFailed) {
    gchar *json = NULL;

    if (isSeg) {
      // Seg model: 4 split outputs, each with own quantization
      QuantParams qp[MODEL_SEG_NUM_OUTPUTS];
      for (guint i = 0; i < MODEL_SEG_NUM_OUTPUTS; i++) {
        if (!extractQuantParams(buffer, i, qp[i])) {
          log_error("No quant meta for output tensor %u — requires updated NNStreamer\n", i);
          return;
        }
        log_info("Output[%u] quant: scale=%g zero_point=%" G_GINT64_FORMAT "\n",
                 i, qp[i].scale, qp[i].zeroPoint);
      }
      json = g_strdup_printf(HAL_SEG_CONFIG_FMT,
                              qp[0].scale, qp[0].zeroPoint,   // classes
                              qp[1].scale, qp[1].zeroPoint,   // protos
                              qp[2].scale, qp[2].zeroPoint,   // bbox
                              qp[3].scale, qp[3].zeroPoint);  // coefficients
    } else {
      // Det model: single output
      QuantParams qp;
      if (!extractQuantParams(buffer, 0, qp)) {
        log_error("No quant meta on output buffer — requires updated NNStreamer\n");
        return;
      }
      log_info("Quant meta: scale=%g zero_point=%" G_GINT64_FORMAT "\n",
               qp.scale, qp.zeroPoint);
      json = g_strdup_printf(HAL_DET_CONFIG_FMT, qp.scale, qp.zeroPoint);
    }

    log_info("HAL config JSON: %s\n", json);

    hal_decoder_params *params = hal_decoder_params_new();
    int jsonRet = hal_decoder_params_set_config_json(params, json, 0);
    if (jsonRet != 0) {
      log_error("hal_decoder_params_set_config_json failed (%d)\n", jsonRet);
    }
    hal_decoder_params_set_score_threshold(params, CONF_THRESHOLD);
    hal_decoder_params_set_iou_threshold(params, NMS_IOU_THRESHOLD);
    hal_decoder_params_set_nms(params, HAL_NMS_CLASS_AGNOSTIC);
    app->decoder = hal_decoder_new(params);
    hal_decoder_params_free(params);
    g_free(json);

    if (!app->decoder) {
      log_error("Failed to create HAL decoder (config: %s)\n", isSeg ? "seg" : "det");
      app->decoderInitFailed = true;
      return;
    }
    log_info("HAL decoder: created (%s mode)\n", isSeg ? "segmentation" : "detection");
  }

  if (!app->decoder) return;

  // Map all output tensors
  GstMapInfo mapInfos[MODEL_SEG_NUM_OUTPUTS] = {};
  bool mapped[MODEL_SEG_NUM_OUTPUTS] = {};
  for (guint i = 0; i < n_mem; i++) {
    GstMemory *mem = gst_buffer_peek_memory(buffer, i);
    if (!gst_memory_map(mem, &mapInfos[i], GST_MAP_READ)) {
      log_error("Can't map output tensor %u\n", i);
      for (guint j = 0; j < i; j++)
        if (mapped[j]) gst_memory_unmap(gst_buffer_peek_memory(buffer, j), &mapInfos[j]);
      return;
    }
    mapped[i] = true;
  }

  // Create HAL tensors
  hal_tensor *hal_tensors[MODEL_SEG_NUM_OUTPUTS] = {};
  hal_detect_box_list *boxes = NULL;

  // Seg model output shapes (matching NNStreamer output order)
  static const size_t seg_shapes[][4] = {
    {1, NUM_CLASSES, NUM_TOTAL_BOXES, 0},             // [0] classes: [1,80,8400]
    {1, MODEL_SEG_PROTO_DIM, MODEL_SEG_PROTO_DIM,     // [1] protos:  [1,160,160,32]
     MODEL_SEG_NUM_PROTOS},
    {1, NUM_COORDINATES, NUM_TOTAL_BOXES, 0},          // [2] bbox:    [1,4,8400]
    {1, MODEL_SEG_NUM_PROTOS, NUM_TOTAL_BOXES, 0},     // [3] coeffs:  [1,32,8400]
  };
  static const size_t seg_ndims[] = {3, 4, 3, 3};

  // Det model output shape
  static const size_t det_shape[] = {1, MODEL_OUTPUT_WIDTH, NUM_TOTAL_BOXES};

  bool tensorsOk = true;
  for (guint i = 0; i < n_mem; i++) {
    const size_t *shape = isSeg ? seg_shapes[i] : det_shape;
    size_t ndim = isSeg ? seg_ndims[i] : 3;
    hal_tensors[i] = hal_tensor_new(HAL_DTYPE_I8, shape, ndim,
                                     HAL_TENSOR_MEMORY_MEM, NULL);
    if (!hal_tensors[i]) { tensorsOk = false; break; }

    hal_tensor_map *tmap = hal_tensor_map_create(hal_tensors[i]);
    if (tmap) {
      void *dst = hal_tensor_map_data(tmap);
      if (dst) memcpy(dst, mapInfos[i].data, mapInfos[i].size);
      hal_tensor_map_unmap(tmap);
    }
  }

  if (tensorsOk) {
    gettimeofday(&halStart, NULL);

    const hal_tensor *outputs[MODEL_SEG_NUM_OUTPUTS];
    for (guint i = 0; i < n_mem; i++)
      outputs[i] = hal_tensors[i];

    int ret;
    if (isSeg && !app->headless && app->seg.overlay) {
      // Seg display: fused decode + GPU mask rendering
      g_mutex_lock(&app->seg.mutex);
      ret = hal_decoder_draw_masks(app->decoder, app->seg.processor,
                                    outputs, n_mem,
                                    app->seg.overlay, &boxes);
      app->seg.overlayReady = (ret == 0);
      g_mutex_unlock(&app->seg.mutex);

      gettimeofday(&halEnd, NULL);
      app->timing.halDrawMasks.record(timeDiffMs(halStart, halEnd));
    } else if (isSeg && app->headless) {
      // Seg headless: decode only (metrics, no rendering)
      ret = hal_decoder_decode(app->decoder, outputs, n_mem, &boxes, NULL);

      gettimeofday(&halEnd, NULL);
      app->timing.halDecode.record(timeDiffMs(halStart, halEnd));
    } else {
      // Det mode: unchanged
      ret = hal_decoder_decode(app->decoder, outputs, n_mem, &boxes, NULL);

      gettimeofday(&halEnd, NULL);
      app->timing.halDecode.record(timeDiffMs(halStart, halEnd));
    }

    if (ret == 0 && boxes) {
      size_t num_dets = hal_detect_box_list_len(boxes);
      app->results.clear();

      for (size_t d = 0; d < num_dets; d++) {
        hal_detect_box box;
        if (hal_detect_box_list_get(boxes, d, &box) == 0) {
          float cx = box.xmin * MODEL_INPUT_SIZE;
          float cy = box.ymin * MODEL_INPUT_SIZE;
          float bw = (box.xmax - box.xmin) * MODEL_INPUT_SIZE;
          float bh = (box.ymax - box.ymin) * MODEL_INPUT_SIZE;

          float px = (cx - app->letterbox.padX) / app->letterbox.scale;
          float py = (cy - app->letterbox.padY) / app->letterbox.scale;
          bw /= app->letterbox.scale;
          bh /= app->letterbox.scale;

          app->results.push_back({(int)box.label, box.score, px, py, bw, bh});
        }
      }

      if (!app->results.empty()) {
        app->timing.framesWithDetections++;
        app->timing.totalDetections += app->results.size();
      }
    }

    if (boxes)
      hal_detect_box_list_free(boxes);
  }

  // Free HAL tensors and unmap GstMemory
  for (guint i = 0; i < n_mem; i++) {
    if (hal_tensors[i]) hal_tensor_free(hal_tensors[i]);
    if (mapped[i]) gst_memory_unmap(gst_buffer_peek_memory(buffer, i), &mapInfos[i]);
  }

  struct timeval postEnd;
  gettimeofday(&postEnd, NULL);
  app->timing.postprocTotal.record(timeDiffMs(callbackStart, postEnd));

  // PTS-correlated E2E
  struct timeval ptsStart;
  if (app->ptsTracker.consumeStart(buffer, ptsStart)) {
    app->timing.e2ePipeline.record(timeDiffMs(ptsStart, postEnd));
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

  if (app->seg.segMode && app->seg.overlay) {
    // Seg mode: composite PBO overlay (HAL already drew masks + boxes)
    g_mutex_lock(&app->seg.mutex);
    if (app->seg.overlayReady) {
      hal_tensor_map *tmap = hal_tensor_map_create(app->seg.overlay);
      if (tmap) {
        void *pixels = hal_tensor_map_data(tmap);
        if (pixels) {
          int w = app->seg.overlayWidth;
          int h = app->seg.overlayHeight;
          int stride = w * 4;  // RGBA = 4 bytes per pixel

          // HAL writes RGBA, Cairo expects ARGB32 (BGRA on little-endian).
          // In-place RGBA→BGRA byte swizzle for the overlay.
          uint8_t *p = (uint8_t *)pixels;
          for (int i = 0; i < w * h; i++) {
            uint8_t r = p[0], g = p[1], b = p[2], a = p[3];
            p[0] = b; p[1] = g; p[2] = r; p[3] = a;
            p += 4;
          }

          cairo_surface_t *surf = cairo_image_surface_create_for_data(
              (unsigned char *)pixels, CAIRO_FORMAT_ARGB32, w, h, stride);
          if (cairo_surface_status(surf) == CAIRO_STATUS_SUCCESS) {
            cairo_set_source_surface(cr, surf, 0, 0);
            cairo_paint(cr);
          }
          cairo_surface_destroy(surf);
        }
        hal_tensor_map_unmap(tmap);
      }
      app->seg.overlayReady = false;
    }
    g_mutex_unlock(&app->seg.mutex);
  } else if (!app->results.empty()) {
    // Det mode: draw bounding boxes via Cairo (unchanged)
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

  gettimeofday(&endTime, NULL);
  app->timing.cairoDraw.record(timeDiffMs(startTime, endTime));
}


/* ─── Main ────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
  ParsedArgs pargs;
  pargs.camera = DEFAULT_CAMERA_DEVICE;

  uint32_t flags = ARG_MODEL | ARG_CAMERA | ARG_VIDEO | ARG_IMAGE |
                   ARG_HEADLESS | ARG_INSTRUMENTED | ARG_NUM_FRAMES | ARG_SEG;

  int ret = parseArgs(argc, argv, flags,
      "YOLOv8n 640x640 for i.MX 8M Plus — EdgeFirst Optimized", pargs);
  if (ret != 0) return ret > 0 ? 0 : 1;

  if (pargs.model.empty()) {
    log_error("Provide model path with -m\n");
    return 1;
  }

  gst_init(&argc, &argv);

  // Letterbox
  LetterboxParams lb = calculateLetterbox(SOURCE_WIDTH, SOURCE_HEIGHT);

  log_info("YOLOv8n 640x640 for i.MX 8M Plus — EdgeFirst Optimized\n");
  log_info("Model: %s\n", pargs.model.c_str());
  if (!pargs.image.empty()) {
    log_info("Input: image (%s)\n", pargs.image.c_str());
  } else if (!pargs.video.empty()) {
    log_info("Input: video (%s)\n", pargs.video.c_str());
  } else {
    log_info("Input: camera (%s)\n", pargs.camera.c_str());
  }
  if (pargs.numFrames > 0) {
    log_info("Frames: %d\n", pargs.numFrames);
  }
  log_info("Mode: %s\n", pargs.headless ? "headless" : "display");
  log_info("Framework: tensorflow2-lite (V2 invoke + DMA-BUF)\n");
  log_info("CameraAdaptor: rgba (NPU handles RGBA->RGB + UINT8->INT8)\n");
  log_info("Post-processing: EdgeFirst HAL (quantized NMS)\n");
  if (pargs.segmentation)
    log_info("Segmentation: enabled (GPU-accelerated mask overlay)\n");
  log_info("Letterbox: %dx%d -> scale %.4f -> %dx%d + pad L=%d R=%d T=%d B=%d\n",
           SOURCE_WIDTH, SOURCE_HEIGHT, lb.scale, lb.scaledW, lb.scaledH,
           lb.padX, lb.padRight, lb.padY, lb.padBottom);

  // Build EdgeFirst optimized pipeline
  InputSource srcType = determineInputSource(pargs, false);
  char *srcStr = buildSourceElement(srcType, pargs);

  // NN branch (always present)
  char *nnBranch = g_strdup_printf(
      "queue name=thread-nn leaky=2 max-size-buffers=2 ! "
      "imxvideoconvert_g2d name=g2d_scale keep-ratio=true ! "
      "video/x-raw,width=%d,height=%d,format=RGBA ! "
      "tensor_converter name=tconv ! "
      "tensor_filter name=tfilter framework=tensorflow2-lite model=%s "
      "custom=Delegate:External,ExtDelegateLib:libvx_delegate.so,CameraAdaptor:rgba,DmaBuf:true latency=1 ! "
      "tensor_sink name=inferenceOutput",
      MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, pargs.model.c_str());

  char *pipelineStr;
  if (pargs.headless) {
    pipelineStr = g_strdup_printf("%s ! %s", srcStr, nnBranch);
  } else {
    pipelineStr = g_strdup_printf(
        "%s ! tee name=t "
        "t. ! %s "
        "t. ! queue name=thread-img max-size-buffers=2 ! "
        "imxvideoconvert_g2d ! "
        "cairooverlay name=cairo ! "
        "waylandsink sync=%s",
        srcStr, nnBranch,
        (!pargs.video.empty() || !pargs.image.empty()) ? "true" : "false");
  }
  g_free(srcStr);
  g_free(nnBranch);

  log_info("Pipeline: %s\n\n", pipelineStr);

  // Initialize app
  AppData app = {};
  app.letterbox = lb;
  app.headless = pargs.headless;
  app.instrumented = pargs.instrumented;
  app.numFrames = pargs.numFrames;
  app.timing.g2dScale.reset();
  app.timing.tensorConv.reset();
  app.timing.preprocTotal.reset();
  app.timing.inference.reset();
  app.timing.halDecode.reset();
  app.timing.halDrawMasks.reset();
  app.timing.postprocTotal.reset();
  app.timing.cairoDraw.reset();
  app.timing.e2ePipeline.reset();
  app.throughput.reset();
  app.ptsTracker.init();

  // Initialize seg context if --seg requested
  if (pargs.segmentation)
    initSegContext(app.seg, SOURCE_WIDTH, SOURCE_HEIGHT);

  bool startedOnce = false;
  app.busCtx.playing = &app.playing;
  app.busCtx.startedOnce = &startedOnce;
  app.busCtx.videoLoop = !pargs.video.empty() && pargs.image.empty();
  app.busCtx.videoRate = 1.0;
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

  app.tensorFilter = gst_bin_get_by_name(GST_BIN(app.gstPipeline), "tfilter");

  // Connect cairo overlay for display mode
  if (!pargs.headless) {
    GstElement *cairo = gst_bin_get_by_name(GST_BIN(app.gstPipeline), "cairo");
    if (cairo) {
      g_signal_connect(cairo, "draw", G_CALLBACK(drawCallback), &app);
      gst_object_unref(cairo);
    }
  }

  // Install probes for EdgeFirst pipeline (only G2D + tconv)
  GstElement *g2d = gst_bin_get_by_name(GST_BIN(app.gstPipeline), "g2d_scale");
  if (g2d) {
    GstPad *sinkPad = gst_element_get_static_pad(g2d, "sink");
    GstPad *srcPad = gst_element_get_static_pad(g2d, "src");
    if (sinkPad) {
      gst_pad_add_probe(sinkPad, GST_PAD_PROBE_TYPE_BUFFER, g2dSinkProbe, &app, NULL);
      gst_object_unref(sinkPad);
    }
    if (srcPad) {
      gst_pad_add_probe(srcPad, GST_PAD_PROBE_TYPE_BUFFER, g2dSrcProbe, &app, NULL);
      gst_object_unref(srcPad);
    }
    gst_object_unref(g2d);
  }

  GstElement *tconv = gst_bin_get_by_name(GST_BIN(app.gstPipeline), "tconv");
  if (tconv) {
    GstPad *srcPad = gst_element_get_static_pad(tconv, "src");
    if (srcPad) {
      gst_pad_add_probe(srcPad, GST_PAD_PROBE_TYPE_BUFFER, tensorconvSrcProbe, &app, NULL);
      gst_object_unref(srcPad);
    }
    gst_object_unref(tconv);
  }

  g_unix_signal_add(SIGINT, commonSigintHandler, &app.busCtx);

  // Run
  gst_element_set_state(app.gstPipeline, GST_STATE_PLAYING);
  g_main_loop_run(app.loop);

  // Cleanup — tear down pipeline first so waylandsink stats print before ours
  gst_element_set_state(app.gstPipeline, GST_STATE_NULL);
  if (app.instrumented)
    printTimingStatistics(&app);
  if (app.tensorFilter)
    gst_object_unref(app.tensorFilter);
  gst_object_unref(app.bus);
  gst_object_unref(app.gstPipeline);
  g_main_loop_unref(app.loop);
  app.ptsTracker.destroy();
  if (app.decoder)
    hal_decoder_free(app.decoder);
  if (app.seg.processor)
    freeSegContext(app.seg);

  return 0;
}
