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

// HAL JSON template — quantization filled from GstNnsTensorQuantMeta at runtime
static const char *HAL_CONFIG_FMT =
    "{\"outputs\":[{\"decoder\":\"ultralytics\","
    "\"type\":\"detection\","
    "\"shape\":[1,84,8400],"
    "\"quantization\":[%g,%" G_GINT64_FORMAT "],"
    "\"dshape\":[[\"batch\",1],[\"num_features\",84],[\"num_boxes\",8400]]}],"
    "\"nms\":\"class_agnostic\"}";


/* ─── Timing statistics ───────────────────────────────────────────── */

typedef struct {
  // Preprocessing (EdgeFirst: only 2 stages)
  TimingMetric g2dScale;       // G2D resize + NV12→RGBA + letterbox (FUSED, DMA-BUF)
  TimingMetric tensorConv;     // tensor_converter (RGBA → tensor)
  TimingMetric preprocTotal;   // Total preprocessing

  // Inference (includes CameraAdaptor Slice + DataConvert on NPU)
  TimingMetric inference;

  // Post-processing (HAL decoder)
  TimingMetric halDecode;      // HAL decoder (quantized NMS)
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
  printf("\n  POST-PROCESSING (EdgeFirst HAL — quantized NMS, no dequantization)\n");
  printf("  --------------------------------------------------------------------------\n");
  printMetric("HAL decoder (dequant + NMS in one call)",
              "operates on INT8 tensors directly", app->timing.halDecode);
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
  if (n_mem != 1) { log_error("Expected 1 tensor, got %u\n", n_mem); return; }

  // Lazy HAL decoder init — extract quant params from GstNnsTensorQuantMeta
  if (!app->decoder) {
    QuantParams qp;
    if (!extractQuantParams(buffer, 0, qp)) {
      log_error("No quant meta on output buffer — requires updated NNStreamer\n");
      return;
    }
    log_info("Quant meta: scale=%g zero_point=%" G_GINT64_FORMAT "\n",
             qp.scale, qp.zeroPoint);

    gchar *json = g_strdup_printf(HAL_CONFIG_FMT, qp.scale, qp.zeroPoint);
    hal_decoder_params *params = hal_decoder_params_new();
    hal_decoder_params_set_config_json(params, json, 0);
    hal_decoder_params_set_score_threshold(params, CONF_THRESHOLD);
    hal_decoder_params_set_iou_threshold(params, NMS_IOU_THRESHOLD);
    hal_decoder_params_set_nms(params, HAL_NMS_CLASS_AGNOSTIC);
    app->decoder = hal_decoder_new(params);
    hal_decoder_params_free(params);
    g_free(json);

    if (!app->decoder) {
      log_error("Failed to create HAL decoder\n");
      return;
    }
    log_info("HAL decoder: created successfully\n");
  }

  // Map output tensor
  // NOTE: With DMA-BUF V2 path, we must keep the mapping valid while reading
  GstMemory *mem = gst_buffer_peek_memory(buffer, 0);
  GstMapInfo info;
  if (!gst_memory_map(mem, &info, GST_MAP_READ)) {
    log_error("Can't map output tensor\n");
    return;
  }

  // Create HAL tensor — single [1, 84, 8400] INT8 tensor
  size_t shape[] = {1, MODEL_OUTPUT_WIDTH, NUM_TOTAL_BOXES};
  hal_tensor *hal_out = hal_tensor_new(HAL_DTYPE_I8, shape, 3,
                                        HAL_TENSOR_MEMORY_MEM, NULL);
  hal_detect_box_list *boxes = NULL;

  if (hal_out) {
    hal_tensor_map *tmap = hal_tensor_map_create(hal_out);
    if (tmap) {
      void *dst = hal_tensor_map_data(tmap);
      if (dst)
        memcpy(dst, info.data, info.size);
      hal_tensor_map_unmap(tmap);
    }

    gettimeofday(&halStart, NULL);

    const hal_tensor *outputs[] = {hal_out};
    int ret = hal_decoder_decode(app->decoder, outputs, 1, &boxes, NULL);

    gettimeofday(&halEnd, NULL);
    app->timing.halDecode.record(timeDiffMs(halStart, halEnd));

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
    hal_tensor_free(hal_out);
  }

  // Unmap output tensor — all reads done
  gst_memory_unmap(mem, &info);

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
  if (app->results.empty()) return;

  struct timeval startTime, endTime;
  gettimeofday(&startTime, NULL);

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

  gettimeofday(&endTime, NULL);
  app->timing.cairoDraw.record(timeDiffMs(startTime, endTime));
}


/* ─── Main ────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
  ParsedArgs pargs;
  pargs.camera = DEFAULT_CAMERA_DEVICE;

  uint32_t flags = ARG_MODEL | ARG_CAMERA | ARG_VIDEO | ARG_IMAGE |
                   ARG_HEADLESS | ARG_INSTRUMENTED | ARG_NUM_FRAMES;

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
        "imxvideoconvert_g2d ! video/x-raw,format=RGBA ! "
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
  app.timing.postprocTotal.reset();
  app.timing.cairoDraw.reset();
  app.timing.e2ePipeline.reset();
  app.throughput.reset();
  app.ptsTracker.init();

  bool startedOnce = false;
  app.busCtx.playing = &app.playing;
  app.busCtx.startedOnce = &startedOnce;
  app.busCtx.videoLoop = !pargs.video.empty() && pargs.image.empty();
  app.busCtx.videoRate = 1.0;
  app.busCtx.printTiming = printTimingStatistics;
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

  // Print timing statistics (guard prevents double print)
  if (app.instrumented)
    printTimingStatistics(&app);


  // Cleanup
  gst_element_set_state(app.gstPipeline, GST_STATE_NULL);
  if (app.tensorFilter)
    gst_object_unref(app.tensorFilter);
  gst_object_unref(app.bus);
  gst_object_unref(app.gstPipeline);
  g_main_loop_unref(app.loop);
  app.ptsTracker.destroy();
  if (app.decoder)
    hal_decoder_free(app.decoder);

  return 0;
}
