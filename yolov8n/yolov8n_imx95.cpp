/**
 * Copyright 2025 NXP
 * Copyright 2026 EdgeFirst AI (Au-Zone Technologies)
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * YOLOv8n 640x640 Camera Demo for i.MX 95 — EdgeFirst CameraAdaptor + HAL
 *
 * PREPROCESSING (1 element — edgefirstcameraadaptor):
 *   - NV12 → RGB color conversion (G2D/GPU hardware)
 *   - Resize + letterbox with grey fill (G2D/GPU hardware)
 *   - uint8 → int8 quantization shift (NEON XOR 0x80)
 *
 * INFERENCE: Neutron NPU via libneutron_delegate.so
 *
 * POST-PROCESSING: EdgeFirst HAL decoder (quantized NMS, no dequantization)
 *   - Model metadata parsed from TFLite zipfile by HAL at runtime
 *   - No external config files needed
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


/* ─── Timing statistics ───────────────────────────────────────────── */

typedef struct {
  // Preprocessing (edgefirstcameraadaptor — fused HAL pipeline)
  TimingMetric preproc;

  // Inference
  TimingMetric inference;

  // Post-processing (HAL decoder)
  TimingMetric halDecode;      // HAL decoder (dequant + NMS in one call)
  TimingMetric postprocTotal;

  // Rendering
  TimingMetric cairoDraw;

  // End-to-end
  TimingMetric e2e;
  TimingMetric e2ePipeline;

  // Timestamps for pad probes
  struct timeval preprocStart;
  struct timeval lastFrameTime;
  bool firstFrame;

  // PTS-correlated E2E
  std::map<GstClockTime, struct timeval> nnPipelineStart;
  GMutex e2eMutex;

  // Detection statistics
  int totalDetections;
  int framesWithDetections;
} TimingStats;


/* ─── Application data ────────────────────────────────────────────── */

struct DetectionResult {
  int classId;
  float score;
  float x, y, w, h;  // Source-space pixel coordinates
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
  bool videoLoop;
  bool startedOnce;

  // HAL decoder
  hal_decoder *decoder;

  // CameraAdaptor element (for letterbox query)
  GstElement *cameraadaptor;
} AppData;


/* ─── Pad probes ──────────────────────────────────────────────────── */

static GstPadProbeReturn preprocSinkProbe(GstPad *, GstPadProbeInfo *info, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->timing.preprocStart, NULL);

  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (buffer) {
    GstClockTime pts = GST_BUFFER_PTS(buffer);
    if (GST_CLOCK_TIME_IS_VALID(pts)) {
      g_mutex_lock(&app->timing.e2eMutex);
      app->timing.nnPipelineStart[pts] = app->timing.preprocStart;
      while (app->timing.nnPipelineStart.size() > 100)
        app->timing.nnPipelineStart.erase(app->timing.nnPipelineStart.begin());
      g_mutex_unlock(&app->timing.e2eMutex);
    }
  }
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn preprocSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  struct timeval end;
  gettimeofday(&end, NULL);
  app->timing.preproc.record(timeDiffMs(app->timing.preprocStart, end));
  return GST_PAD_PROBE_OK;
}


/* ─── Timing report ───────────────────────────────────────────────── */

static void printTimingStatistics(AppData *app)
{
  printf("\n");
  printf("==============================================================================\n");
  printf("  EDGEFIRST CAMERAADAPTOR + HAL — i.MX 95 Neutron NPU\n");
  printf("==============================================================================\n");

  if (app->instrumented) {
    printf("\n  PREPROCESSING (edgefirstcameraadaptor — fused HAL pipeline)\n");
    printf("  --------------------------------------------------------------------------\n");
    printf("     NV12 -> RGB + resize + letterbox (G2D/GPU) + uint8->int8 (NEON)\n");
    if (app->timing.preproc.count > 0) {
      printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d frames]\n",
             app->timing.preproc.avg(), app->timing.preproc.minMs,
             app->timing.preproc.maxMs, app->timing.preproc.count);
    }
  }

  printf("\n==============================================================================\n");
  printf("\n  INFERENCE (Neutron NPU)\n");
  printf("  --------------------------------------------------------------------------\n");
  if (app->timing.inference.count > 0) {
    printf("     tensor_filter with libneutron_delegate.so\n");
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

  if (app->timing.e2ePipeline.count > 0) {
    printf("  E2E NN Pipeline (PTS-correlated):\n");
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d frames]\n",
           app->timing.e2ePipeline.avg(), app->timing.e2ePipeline.minMs,
           app->timing.e2ePipeline.maxMs, app->timing.e2ePipeline.count);
  }

  if (app->timing.e2e.count > 0) {
    double avgMs = app->timing.e2e.avg();
    printf("\n  Frame Throughput:\n");
    printf("     Average: %7.3f ms (%5.1f FPS)  |  Frames: %d\n",
           avgMs, 1000.0 / avgMs, app->timing.e2e.count);
  }

  printf("\n  Detection Statistics:\n");
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
      g_error_free(err);
      g_free(debugInfo);
      g_main_loop_quit(app->loop);
      break;
    }
    case GST_MESSAGE_EOS:
      if (app->videoLoop) {
        gst_element_seek_simple(app->gstPipeline, GST_FORMAT_TIME,
                                (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT), 0);
      } else {
        log_info("End-Of-Stream.\n");
        printTimingStatistics(app);
        g_main_loop_quit(app->loop);
      }
      break;
    case GST_MESSAGE_STATE_CHANGED: {
      GstState oldState, newState, pendingState;
      gst_message_parse_state_changed(message, &oldState, &newState, &pendingState);
      if (GST_MESSAGE_SRC(message) == GST_OBJECT(app->gstPipeline)) {
        if (!app->startedOnce) {
          log_info("Pipeline: %s -> %s\n",
                   gst_element_state_get_name(oldState),
                   gst_element_state_get_name(newState));
        }
        if (newState == GST_STATE_PLAYING)
          app->startedOnce = true;
        app->playing = (newState == GST_STATE_PLAYING);
      }
      break;
    }
    default:
      break;
  }
}

static void newDataCallback(GstElement *, GstBuffer *buffer, gpointer user_data)
{
  struct timeval callbackStart, halStart, halEnd;
  gettimeofday(&callbackStart, NULL);

  AppData *app = (AppData *)user_data;

  // Query letterbox from cameraadaptor on first frame (caps negotiated)
  if (app->timing.firstFrame && app->cameraadaptor) {
    gfloat scale = 0.0f;
    gint top = 0, left = 0;
    g_object_get(app->cameraadaptor, "letterbox-scale", &scale,
                 "letterbox-top", &top, "letterbox-left", &left, NULL);
    app->letterbox.scale = scale;
    app->letterbox.padX = left;
    app->letterbox.padY = top;
    log_info("Letterbox (from cameraadaptor): scale=%.4f top=%d left=%d\n",
             scale, top, left);
  }

  // Frame-to-frame interval
  if (!app->timing.firstFrame)
    app->timing.e2e.record(timeDiffMs(app->timing.lastFrameTime, callbackStart));
  app->timing.lastFrameTime = callbackStart;
  app->timing.firstFrame = false;

  // Inference latency
  if (app->tensorFilter) {
    gint64 latencyUs = 0;
    g_object_get(app->tensorFilter, "latency", &latencyUs, NULL);
    if (latencyUs > 0)
      app->timing.inference.record(latencyUs / 1000.0);
  }

  // Validate buffer
  if (!GST_IS_BUFFER(buffer)) { log_error("Invalid buffer\n"); return; }
  guint n_mem = gst_buffer_n_memory(buffer);
  if (n_mem != 1) { log_error("Expected 1 tensor, got %u\n", n_mem); return; }

  // Map output tensor
  GstMemory *mem = gst_buffer_peek_memory(buffer, 0);
  GstMapInfo info;
  if (!gst_memory_map(mem, &info, GST_MAP_READ)) {
    log_error("Can't map output tensor\n");
    return;
  }

  // Create HAL tensor from mapped data
  // Neutron outputs a single [1, 84, 8400] INT8 tensor
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

    // Decode with HAL — handles dequant + NMS in one call
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
          // HAL returns normalized [0,1] → pixel → remove letterbox → source coords
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

  gst_memory_unmap(mem, &info);

  // Record total post-processing time
  struct timeval postEnd;
  gettimeofday(&postEnd, NULL);
  app->timing.postprocTotal.record(timeDiffMs(callbackStart, postEnd));

  // PTS-correlated E2E
  GstClockTime pts = GST_BUFFER_PTS(buffer);
  if (GST_CLOCK_TIME_IS_VALID(pts)) {
    g_mutex_lock(&app->timing.e2eMutex);
    auto it = app->timing.nnPipelineStart.find(pts);
    if (it != app->timing.nnPipelineStart.end()) {
      app->timing.e2ePipeline.record(timeDiffMs(it->second, postEnd));
      app->timing.nnPipelineStart.erase(it);
    }
    g_mutex_unlock(&app->timing.e2eMutex);
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
    const char *name = (det.classId < NUM_CLASSES) ? cocoClassNames[det.classId] : "?";
    char label[64];
    snprintf(label, sizeof(label), "%s %.0f%%", name, det.score * 100);
    cairo_show_text(cr, label);
  }
  cairo_stroke(cr);

  gettimeofday(&endTime, NULL);
  app->timing.cairoDraw.record(timeDiffMs(startTime, endTime));
}


/* ─── Install pad probes ──────────────────────────────────────────── */

static void installProbes(AppData *app)
{
  if (!app->cameraadaptor) return;

  GstPad *sinkPad = gst_element_get_static_pad(app->cameraadaptor, "sink");
  GstPad *srcPad = gst_element_get_static_pad(app->cameraadaptor, "src");
  if (sinkPad) {
    gst_pad_add_probe(sinkPad, GST_PAD_PROBE_TYPE_BUFFER, preprocSinkProbe, app, NULL);
    gst_object_unref(sinkPad);
  }
  if (srcPad) {
    gst_pad_add_probe(srcPad, GST_PAD_PROBE_TYPE_BUFFER, preprocSrcProbe, app, NULL);
    gst_object_unref(srcPad);
  }
}


/* ─── Command line ────────────────────────────────────────────────── */

static int parseArgs(int argc, char **argv, std::string &model,
                     std::string &video, std::string &image,
                     bool &headless, bool &instrumented)
{
  static struct option longOptions[] = {
    {"help",         no_argument,       0, 'h'},
    {"model",        required_argument, 0, 'm'},
    {"video",        required_argument, 0, 'v'},
    {"image",        required_argument, 0, 'i'},
    {"headless",     no_argument,       0, 'H'},
    {"instrumented", no_argument,       0, 'I'},
    {0, 0, 0, 0}
  };

  int c;
  while ((c = getopt_long(argc, argv, "hm:v:i:HI", longOptions, NULL)) != -1) {
    switch (c) {
      case 'h':
        std::cout
            << "YOLOv8n 640x640 for i.MX 95 — EdgeFirst CameraAdaptor + HAL\n\n"
            << "Usage: " << argv[0] << " -m MODEL [options]\n\n"
            << "Options:\n"
            << "  -m, --model PATH        Model file (_converted.tflite) [required]\n"
            << "  -v, --video FILE        Video file input (H.264 MP4)\n"
            << "  -i, --image FILE        Static image input (JPEG)\n"
            << "  -I, --instrumented      Enable detailed timing breakdown\n"
            << "  -H, --headless          No display output\n"
            << "  -h, --help              Show this help\n";
        return 1;
      case 'm': model = optarg; break;
      case 'v': video = optarg; break;
      case 'i': image = optarg; break;
      case 'H': headless = true; break;
      case 'I': instrumented = true; break;
    }
  }
  return 0;
}


/* ─── Main ────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
  std::string model;
  std::string video;
  std::string image;
  bool headless = false;
  bool instrumented = false;

  int ret = parseArgs(argc, argv, model, video, image, headless, instrumented);
  if (ret != 0) return ret > 0 ? 0 : 1;

  if (model.empty()) {
    log_error("Provide model path with -m\n");
    return 1;
  }

  // Auto-set i.MX 95 environment
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

  gst_init(&argc, &argv);

  // Initialize HAL decoder — YOLOv8n combined [84, 8400] INT8 output
  hal_decoder_params params = hal_decoder_params_default();
  params.config_json =
      "{\"outputs\":[{\"decoder\":\"ultralytics\","
      "\"type\":\"detection\","
      "\"shape\":[1,84,8400],"
      "\"quantization\":[0.00390632,-128],"
      "\"dshape\":[[\"batch\",1],[\"num_features\",84],[\"num_boxes\",8400]]}],"
      "\"nms\":\"class_agnostic\"}";
  params.score_threshold = CONF_THRESHOLD;
  params.iou_threshold = NMS_IOU_THRESHOLD;
  params.nms = HAL_NMS_CLASS_AGNOSTIC;

  hal_decoder *decoder = hal_decoder_new(&params);
  if (!decoder) {
    log_error("Failed to create HAL decoder\n");
    return 1;
  }
  log_info("HAL decoder: ultralytics detection [84,8400] INT8\n");

  log_info("YOLOv8n 640x640 for i.MX 95 — EdgeFirst CameraAdaptor + HAL\n");
  log_info("Model: %s\n", model.c_str());
  if (!image.empty()) {
    log_info("Input: image (%s)\n", image.c_str());
  } else if (!video.empty()) {
    log_info("Input: video (%s)\n", video.c_str());
  } else {
    log_info("Input: camera (libcamerasrc)\n");
  }
  if (headless) log_info("Mode: headless (no display)\n");

  // Build source element
  char *sourceStr;
  const char *syncMode = "false";
  if (!image.empty()) {
    sourceStr = g_strdup_printf(
        "filesrc location=%s ! jpegdec ! imxvideoconvert_g2d ! "
        "video/x-raw,width=%d,height=%d ! imagefreeze",
        image.c_str(), SOURCE_WIDTH, SOURCE_HEIGHT);
    syncMode = "true";
  } else if (!video.empty()) {
    sourceStr = g_strdup_printf(
        "filesrc location=%s ! qtdemux ! h264parse ! v4l2h264dec",
        video.c_str());
    syncMode = "true";
  } else {
    sourceStr = g_strdup_printf(
        "libcamerasrc ! video/x-raw,format=NV12,width=%d,height=%d",
        SOURCE_WIDTH, SOURCE_HEIGHT);
  }

  // Build NN processing branch
  // edgefirstcameraadaptor: fused NV12→RGB + resize + letterbox (G2D/GPU)
  //   + uint8→int8 quantization shift (NEON)
  char *nnBranch = g_strdup_printf(
      "queue name=thread-nn leaky=2 max-size-buffers=2 ! "
      "edgefirstcameraadaptor name=preproc model-width=%d model-height=%d "
      "model-dtype=int8 model-layout=hwc letterbox=true ! "
      "tensor_filter name=tfilter framework=tensorflow-lite model=%s "
      "custom=Delegate:External,ExtDelegateLib:libneutron_delegate.so latency=1 ! "
      "tensor_sink name=inferenceOutput",
      MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
      model.c_str());

  // Build full pipeline
  char *pipelineStr;
  if (headless) {
    pipelineStr = g_strdup_printf("%s ! %s", sourceStr, nnBranch);
  } else {
    pipelineStr = g_strdup_printf(
        "%s ! tee name=t "
        "t. ! %s "
        "t. ! queue name=thread-img max-size-buffers=2 ! "
        "imxvideoconvert_g2d ! "
        "cairooverlay name=cairo ! "
        "waylandsink sync=%s",
        sourceStr, nnBranch, syncMode);
  }
  g_free(sourceStr);
  g_free(nnBranch);

  log_info("Pipeline: %s\n\n", pipelineStr);

  // Initialize app
  AppData app = {};
  app.headless = headless;
  app.instrumented = instrumented;
  app.decoder = decoder;
  app.videoLoop = !video.empty() && image.empty();
  app.startedOnce = false;
  app.timing.preproc.reset();
  app.timing.inference.reset();
  app.timing.halDecode.reset();
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
    hal_decoder_free(decoder);
    return 1;
  }

  // Connect signals
  app.bus = gst_element_get_bus(app.gstPipeline);
  gst_bus_add_signal_watch(app.bus);
  g_signal_connect(app.bus, "message", G_CALLBACK(busCallback), &app);

  GstElement *tsink = gst_bin_get_by_name(GST_BIN(app.gstPipeline), "inferenceOutput");
  g_signal_connect(tsink, "new-data", G_CALLBACK(newDataCallback), &app);
  gst_object_unref(tsink);

  if (!headless) {
    GstElement *cairo = gst_bin_get_by_name(GST_BIN(app.gstPipeline), "cairo");
    if (cairo) {
      g_signal_connect(cairo, "draw", G_CALLBACK(drawCallback), &app);
      gst_object_unref(cairo);
    }
  }

  app.tensorFilter = gst_bin_get_by_name(GST_BIN(app.gstPipeline), "tfilter");
  app.cameraadaptor = gst_bin_get_by_name(GST_BIN(app.gstPipeline), "preproc");

  // Install probes (always — needed for E2E PTS correlation)
  installProbes(&app);

  g_unix_signal_add(SIGINT, sigintHandler, &app);

  // Run
  gst_element_set_state(app.gstPipeline, GST_STATE_PLAYING);
  g_main_loop_run(app.loop);

  // Cleanup
  gst_element_set_state(app.gstPipeline, GST_STATE_NULL);
  if (app.tensorFilter)
    gst_object_unref(app.tensorFilter);
  if (app.cameraadaptor)
    gst_object_unref(app.cameraadaptor);
  gst_object_unref(app.bus);
  gst_object_unref(app.gstPipeline);
  g_main_loop_unref(app.loop);
  g_mutex_clear(&app.timing.e2eMutex);
  hal_decoder_free(decoder);

  return 0;
}
