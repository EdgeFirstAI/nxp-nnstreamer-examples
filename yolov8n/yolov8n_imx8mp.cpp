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

  // End-to-end
  TimingMetric e2e;
  TimingMetric e2ePipeline;

  // Timestamps
  struct timeval g2dStart;
  struct timeval g2dEnd;
  struct timeval tensorconvEnd;
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
  bool videoLoop;
  bool startedOnce;

  // HAL decoder
  hal_decoder *decoder;
} AppData;


/* ─── Pad probes ──────────────────────────────────────────────────── */

static GstPadProbeReturn g2dSinkProbe(GstPad *, GstPadProbeInfo *info, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->timing.g2dStart, NULL);

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

static GstPadProbeReturn tensorconvSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->timing.tensorconvEnd, NULL);
  app->timing.tensorConv.record(timeDiffMs(app->timing.g2dEnd, app->timing.tensorconvEnd));
  app->timing.preprocTotal.record(timeDiffMs(app->timing.g2dStart, app->timing.tensorconvEnd));
  return GST_PAD_PROBE_OK;
}


/* ─── Timing report ───────────────────────────────────────────────── */

static void printTimingStatistics(AppData *app)
{
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
      if (GST_MESSAGE_SRC(message) == GST_OBJECT(app->gstPipeline) &&
          oldState != newState) {
        if (!app->startedOnce) {
          log_info("Pipeline: %s -> %s\n",
                   gst_element_state_get_name(oldState),
                   gst_element_state_get_name(newState));
        }
        if (newState == GST_STATE_PLAYING) {
          app->playing = true;
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
  struct timeval callbackStart, halStart, halEnd;
  gettimeofday(&callbackStart, NULL);

  AppData *app = (AppData *)user_data;

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


/* ─── Command line ────────────────────────────────────────────────── */

static int parseArgs(int argc, char **argv, std::string &model, std::string &camera,
                     std::string &video, std::string &image, int &numFrames, bool &headless,
                     bool &instrumented)
{
  static struct option longOptions[] = {
    {"help",         no_argument,       0, 'h'},
    {"model",        required_argument, 0, 'm'},
    {"camera",       required_argument, 0, 'c'},
    {"video",        required_argument, 0, 'v'},
    {"image",        required_argument, 0, 'i'},
    {"frames",       required_argument, 0, 'n'},
    {"headless",     no_argument,       0, 'H'},
    {"instrumented", no_argument,       0, 'I'},
    {0, 0, 0, 0}
  };

  int c;
  while ((c = getopt_long(argc, argv, "hm:c:v:i:n:HI", longOptions, NULL)) != -1) {
    switch (c) {
      case 'h':
        std::cout
            << "YOLOv8n 640x640 for i.MX 8M Plus — EdgeFirst Optimized\n\n"
            << "Usage: " << argv[0] << " -m MODEL [options]\n\n"
            << "Options:\n"
            << "  -m, --model PATH        Model file (.tflite) [required]\n"
            << "  -c, --camera DEVICE     Camera device (default: " << DEFAULT_CAMERA_DEVICE << ")\n"
            << "  -v, --video FILE        Video file input (H.264 MP4)\n"
            << "  -i, --image FILE        Static image input (JPEG)\n"
            << "  -n, --frames N          Stop after N frames (0=infinite, default=0)\n"
            << "  -I, --instrumented      Enable detailed timing breakdown\n"
            << "  -H, --headless          No display output\n"
            << "  -h, --help              Show this help\n\n"
            << "EdgeFirst optimizations:\n"
            << "  * tensorflow2-lite V2 invoke with DMA-BUF zero-copy\n"
            << "  * G2D keep-ratio=true: HW letterbox to delegate DMA-BUF\n"
            << "  * CameraAdaptor:rgba: NPU handles RGBA->RGB + UINT8->INT8\n"
            << "  * EdgeFirst HAL: quantized NMS (no CPU dequantization)\n";
        return 1;
      case 'm': model = optarg; break;
      case 'c': camera = optarg; break;
      case 'v': video = optarg; break;
      case 'i': image = optarg; break;
      case 'n': numFrames = atoi(optarg); break;
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
  std::string camera = DEFAULT_CAMERA_DEVICE;
  std::string video;
  std::string image;
  int numFrames = 0;
  bool headless = false;
  bool instrumented = false;

  int ret = parseArgs(argc, argv, model, camera, video, image, numFrames, headless, instrumented);
  if (ret != 0) return ret > 0 ? 0 : 1;

  if (model.empty()) {
    log_error("Provide model path with -m\n");
    return 1;
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
    log_error("Failed to create HAL decoder: %s\n", strerror(errno));
    return 1;
  }
  log_info("HAL decoder: ultralytics detection [84,8400] INT8\n");

  // Letterbox
  LetterboxParams lb = calculateLetterbox(SOURCE_WIDTH, SOURCE_HEIGHT);

  log_info("YOLOv8n 640x640 for i.MX 8M Plus — EdgeFirst Optimized\n");
  log_info("Model: %s\n", model.c_str());
  if (!image.empty()) {
    log_info("Input: image (%s)\n", image.c_str());
  } else if (!video.empty()) {
    log_info("Input: video (%s)\n", video.c_str());
  } else {
    log_info("Input: camera (%s)\n", camera.c_str());
  }
  if (numFrames > 0) {
    log_info("Frames: %d\n", numFrames);
  }
  log_info("Mode: %s\n", headless ? "headless" : "display");
  log_info("Framework: tensorflow2-lite (V2 invoke + DMA-BUF)\n");
  log_info("CameraAdaptor: rgba (NPU handles RGBA->RGB + UINT8->INT8)\n");
  log_info("Post-processing: EdgeFirst HAL (quantized NMS)\n");
  log_info("Letterbox: %dx%d -> scale %.4f -> %dx%d + pad L=%d R=%d T=%d B=%d\n",
           SOURCE_WIDTH, SOURCE_HEIGHT, lb.scale, lb.scaledW, lb.scaledH,
           lb.padX, lb.padRight, lb.padY, lb.padBottom);

  // Build EdgeFirst optimized pipeline
  // Source element
  std::string sourceStr;
  if (!image.empty()) {
    char *s = g_strdup_printf(
        "filesrc location=%s ! jpegdec ! imxvideoconvert_g2d ! "
        "video/x-raw,width=%d,height=%d ! imagefreeze",
        image.c_str(), SOURCE_WIDTH, SOURCE_HEIGHT);
    sourceStr = s;
    g_free(s);
  } else if (!video.empty()) {
    char *s = g_strdup_printf(
        "filesrc location=%s ! qtdemux ! h264parse ! v4l2h264dec",
        video.c_str());
    sourceStr = s;
    g_free(s);
  } else {
    char *s;
    if (numFrames > 0) {
      s = g_strdup_printf(
          "v4l2src device=%s num-buffers=%d ! video/x-raw,format=NV12,width=%d,height=%d,framerate=30/1",
          camera.c_str(), numFrames, SOURCE_WIDTH, SOURCE_HEIGHT);
    } else {
      s = g_strdup_printf(
          "v4l2src device=%s ! video/x-raw,format=NV12,width=%d,height=%d,framerate=30/1",
          camera.c_str(), SOURCE_WIDTH, SOURCE_HEIGHT);
    }
    sourceStr = s;
    g_free(s);
  }

  // NN branch (always present)
  char *nnBranch = g_strdup_printf(
      "queue name=thread-nn leaky=2 max-size-buffers=2 ! "
      "imxvideoconvert_g2d name=g2d_scale keep-ratio=true ! "
      "video/x-raw,width=%d,height=%d,format=RGBA ! "
      "tensor_converter name=tconv ! "
      "tensor_filter name=tfilter framework=tensorflow2-lite model=%s "
      "custom=Delegate:External,ExtDelegateLib:libvx_delegate.so,CameraAdaptor:rgba,DmaBuf:true latency=1 ! "
      "tensor_sink name=inferenceOutput",
      MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, model.c_str());

  char *pipelineStr;
  if (headless) {
    // Headless: source -> NN branch only
    pipelineStr = g_strdup_printf("%s ! %s", sourceStr.c_str(), nnBranch);
  } else {
    // With display: tee -> NN branch + display branch
    pipelineStr = g_strdup_printf(
        "%s ! tee name=t "
        "t. ! %s "
        "t. ! queue name=thread-img max-size-buffers=2 ! "
        "imxvideoconvert_g2d ! video/x-raw,format=RGBA ! "
        "cairooverlay name=cairo ! "
        "waylandsink sync=%s",
        sourceStr.c_str(), nnBranch,
        (!video.empty() || !image.empty()) ? "true" : "false");
  }
  g_free(nnBranch);

  log_info("Pipeline: %s\n\n", pipelineStr);

  // Initialize app
  AppData app = {};
  app.letterbox = lb;
  app.headless = headless;
  app.instrumented = instrumented;
  app.decoder = decoder;
  app.timing.g2dScale.reset();
  app.timing.tensorConv.reset();
  app.timing.preprocTotal.reset();
  app.timing.inference.reset();
  app.timing.halDecode.reset();
  app.timing.postprocTotal.reset();
  app.timing.cairoDraw.reset();
  app.timing.e2e.reset();
  app.timing.e2ePipeline.reset();
  app.timing.firstFrame = true;
  app.videoLoop = !video.empty() && image.empty();
  app.startedOnce = false;
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

  app.tensorFilter = gst_bin_get_by_name(GST_BIN(app.gstPipeline), "tfilter");

  // Connect cairo overlay for display mode
  if (!headless) {
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
  hal_decoder_free(decoder);

  return 0;
}
