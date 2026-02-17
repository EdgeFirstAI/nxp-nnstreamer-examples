/**
 * Copyright 2025 EdgeFirst AI (Au-Zone Technologies)
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * YOLOv8n 640x640 Demo — Kinara Ara-2 NPU + EdgeFirst CameraAdaptor + HAL
 *
 * Runs on FRDM boards with i.MX 8M Plus or i.MX 95 via PCIe m.2.
 * Uses NNStreamer's ara2 sub-plugin for Kinara NPU inference with
 * EdgeFirst HAL for quantized post-processing.
 *
 * Ara-2 YOLOv8n.dvm output tensors (split format):
 *   [0] scores — uint8  [80, 8400]  (post-sigmoid class confidences)
 *   [1] boxes  — int16  [4, 8400]   (cx, cy, w, h in model pixel space)
 *
 * The HAL decoder handles all dequantization, shape matching, and NMS
 * internally based on an EdgeFirst YAML config file.
 *
 * PREPROCESSING (1 element — edgefirstcameraadaptor):
 *   - NV12 -> RGB color conversion (HAL)
 *   - Resize + letterbox with grey fill (HAL)
 *   - HWC -> CHW layout transpose (HAL planar RGB)
 *   - uint8 -> int8 quantization shift (XOR 0x80)
 *
 * PIPELINE: source → edgefirstcameraadaptor → tensor_filter → tensor_sink
 *
 * POST-PROCESSING: EdgeFirst HAL decoder (quantized NMS)
 */

#include <gst/gst.h>
#include <gst/allocators/gstdmabuf.h>
#include <glib.h>
#include <glib-unix.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

#include "common/yolov8_common.hpp"

#include <edgefirst/hal.h>


#define NUM_OUTPUTS 2  /* scores + boxes tensors */
#define DEFAULT_CAMERA_DEVICE "/dev/video3"


/* --- Application data ---------------------------------------------------- */

struct AppData {
  GstElement *pipeline;
  GMainLoop *loop;
  GstBus *bus;
  gboolean playing;

  TimingMetric preproc;
  TimingMetric inference;
  TimingMetric postproc;
  TimingMetric e2e;
  struct timeval preprocStart;
  struct timeval lastFrameTime;
  bool firstFrame;

  int totalDetections;
  int framesWithDetections;
  int totalFrames;
  int maxFrames;

  GstElement *tensorFilter;
  GstElement *cameraadaptor;
  LetterboxParams letterbox;

  // HAL decoder
  hal_decoder *decoder;

  // Output tensor metadata
  hal_dtype out_dtypes[NUM_OUTPUTS];
  size_t out_shapes[NUM_OUTPUTS][4];
  size_t out_ndims[NUM_OUTPUTS];
  gboolean meta_populated;
};


/* --- Preprocessing timing probes ----------------------------------------- */

static GstPadProbeReturn
preprocStartProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->preprocStart, NULL);
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
preprocEndProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  struct timeval end;
  gettimeofday(&end, NULL);
  app->preproc.record(timeDiffMs(app->preprocStart, end));
  return GST_PAD_PROBE_OK;
}


/* --- Timing report ------------------------------------------------------- */

static void printTiming(AppData *app)
{
  printf("\n");
  printf("==============================================================================\n");
  printf("  KINARA ARA-2 NPU + EDGEFIRST CAMERAADAPTOR + HAL\n");
  printf("==============================================================================\n");

  printf("\n  PREPROCESSING (edgefirstcameraadaptor — fused HAL pipeline)\n");
  printf("  --------------------------------------------------------------------------\n");
  if (app->preproc.count > 0) {
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d frames]\n",
           app->preproc.avg(), app->preproc.minMs, app->preproc.maxMs, app->preproc.count);
  }

  printf("\n  INFERENCE (Ara-2 NPU)\n");
  printf("  --------------------------------------------------------------------------\n");
  if (app->inference.count > 0) {
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d samples]\n",
           app->inference.avg(), app->inference.minMs, app->inference.maxMs, app->inference.count);
  }

  printf("\n  POST-PROCESSING (EdgeFirst HAL — quantized NMS)\n");
  printf("  --------------------------------------------------------------------------\n");
  if (app->postproc.count > 0) {
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d frames]\n",
           app->postproc.avg(), app->postproc.minMs, app->postproc.maxMs, app->postproc.count);
  }

  printf("\n  END-TO-END\n");
  printf("  --------------------------------------------------------------------------\n");
  if (app->e2e.count > 0) {
    double avgMs = app->e2e.avg();
    printf("     Average: %7.3f ms (%5.1f FPS)  |  Frames: %d\n",
           avgMs, 1000.0 / avgMs, app->e2e.count);
  }

  printf("\n  DETECTION STATISTICS\n");
  printf("  --------------------------------------------------------------------------\n");
  printf("     Post-NMS detections:     %6d  (avg %.1f/frame)\n",
         app->totalDetections,
         app->framesWithDetections > 0
             ? (double)app->totalDetections / app->framesWithDetections : 0.0);
  printf("     Frames with detections:  %6d / %d\n",
         app->framesWithDetections, app->totalFrames);
  printf("\n==============================================================================\n");
}


/* --- GStreamer bus callback ----------------------------------------------- */

static void busCallback(GstBus *, GstMessage *msg, gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_ERROR: {
      GError *err = NULL;
      gchar *dbg = NULL;
      gst_message_parse_error(msg, &err, &dbg);
      g_printerr("\nERROR from %s: %s\n", GST_OBJECT_NAME(msg->src), err->message);
      if (dbg) g_printerr("Debug: %s\n", dbg);
      g_error_free(err);
      g_free(dbg);
      g_main_loop_quit(app->loop);
      break;
    }
    case GST_MESSAGE_EOS:
      gst_element_seek_simple(app->pipeline, GST_FORMAT_TIME,
                              (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT), 0);
      break;
    case GST_MESSAGE_STATE_CHANGED:
      if (GST_MESSAGE_SRC(msg) == GST_OBJECT(app->pipeline)) {
        GstState old_s, new_s, pending;
        gst_message_parse_state_changed(msg, &old_s, &new_s, &pending);
        g_print("Pipeline: %s -> %s\n",
                gst_element_state_get_name(old_s),
                gst_element_state_get_name(new_s));
        app->playing = (new_s == GST_STATE_PLAYING);
      }
      break;
    default:
      break;
  }
}


/* --- tensor_sink new-data callback --------------------------------------- */

static void newDataCallback(GstElement *, GstBuffer *buffer, gpointer user_data)
{
  struct timeval startTime, endTime;
  gettimeofday(&startTime, NULL);

  AppData *app = (AppData *)user_data;

  // Query letterbox from cameraadaptor on first frame (caps negotiated)
  if (app->firstFrame && app->cameraadaptor) {
    gfloat scale = 0.0f;
    gint top = 0, left = 0;
    g_object_get(app->cameraadaptor, "letterbox-scale", &scale,
                 "letterbox-top", &top, "letterbox-left", &left, NULL);
    app->letterbox.scale = scale;
    app->letterbox.padX = left;
    app->letterbox.padY = top;
    g_print("Letterbox (from cameraadaptor): scale=%.4f top=%d left=%d\n",
            scale, top, left);
  }

  // Frame-to-frame interval
  if (!app->firstFrame) {
    app->e2e.record(timeDiffMs(app->lastFrameTime, startTime));
  }
  app->lastFrameTime = startTime;
  app->firstFrame = false;

  // Inference latency
  if (app->tensorFilter) {
    gint64 latUs = 0;
    g_object_get(app->tensorFilter, "latency", &latUs, NULL);
    if (latUs > 0)
      app->inference.record(latUs / 1000.0);
  }

  // Validate buffer — expect 2 memory blocks (scores + boxes)
  if (!GST_IS_BUFFER(buffer)) { g_printerr("ERROR: invalid buffer\n"); return; }
  guint n_mem = gst_buffer_n_memory(buffer);
  if (n_mem != NUM_OUTPUTS) {
    g_printerr("ERROR: expected %d tensors, got %u\n", NUM_OUTPUTS, n_mem);
    return;
  }

  // Populate output tensor metadata on first frame.
  // HAL decoder requires 3D shapes with batch dimension [batch, feat, boxes].
  if (!app->meta_populated) {
    // Scores: uint8, [1, 80, 8400]
    app->out_dtypes[0] = HAL_DTYPE_U8;
    app->out_shapes[0][0] = 1;
    app->out_shapes[0][1] = 80;
    app->out_shapes[0][2] = 8400;
    app->out_ndims[0] = 3;

    // Boxes: int16, [1, 4, 8400]
    app->out_dtypes[1] = HAL_DTYPE_I16;
    app->out_shapes[1][0] = 1;
    app->out_shapes[1][1] = 4;
    app->out_shapes[1][2] = 8400;
    app->out_ndims[1] = 3;

    app->meta_populated = TRUE;
    g_print("Output tensor metadata:\n");
    g_print("  [0] scores: U8  shape=[%zu, %zu, %zu]\n",
            app->out_shapes[0][0], app->out_shapes[0][1],
            app->out_shapes[0][2]);
    g_print("  [1] boxes:  I16 shape=[%zu, %zu, %zu]\n\n",
            app->out_shapes[1][0], app->out_shapes[1][1],
            app->out_shapes[1][2]);
  }

  // Wrap output tensors as HAL tensors
  hal_tensor *hal_outputs[NUM_OUTPUTS] = {NULL, NULL};
  GstMapInfo out_maps[NUM_OUTPUTS];
  gboolean out_mapped[NUM_OUTPUTS] = {FALSE, FALSE};
  hal_detect_box_list *boxes = NULL;
  int decode_ret = -1;

  for (int j = 0; j < NUM_OUTPUTS; j++) {
    GstMemory *mem = gst_buffer_peek_memory(buffer, j);

    if (gst_is_dmabuf_memory(mem)) {
      int fd = gst_dmabuf_memory_get_fd(mem);
      hal_outputs[j] = hal_tensor_from_fd(
          app->out_dtypes[j], dup(fd),
          app->out_shapes[j], app->out_ndims[j], NULL);
    } else {
      if (!gst_memory_map(mem, &out_maps[j], GST_MAP_READ)) {
        g_printerr("ERROR: cannot map output tensor %d\n", j);
        goto cleanup;
      }
      out_mapped[j] = TRUE;

      hal_outputs[j] = hal_tensor_new(
          app->out_dtypes[j], app->out_shapes[j],
          app->out_ndims[j], HAL_TENSOR_MEMORY_MEM, NULL);

      if (hal_outputs[j]) {
        hal_tensor_map *tmap = hal_tensor_map_create(hal_outputs[j]);
        if (tmap) {
          void *dst = hal_tensor_map_data(tmap);
          if (dst)
            memcpy(dst, out_maps[j].data, out_maps[j].size);
          hal_tensor_map_unmap(tmap);
        }
      }
    }

    if (!hal_outputs[j]) {
      g_printerr("ERROR: failed to create HAL tensor for output %d\n", j);
      goto cleanup;
    }
  }

  // Decode with HAL
  decode_ret = hal_decoder_decode(
      app->decoder,
      (const hal_tensor *const *)hal_outputs, NUM_OUTPUTS,
      &boxes, NULL);

  if (decode_ret != 0) {
    static bool first_error = true;
    if (first_error) {
      g_printerr("WARNING: HAL decoder_decode failed (ret=%d)\n", decode_ret);
      first_error = false;
    }
    goto cleanup;
  }

  // Process detections
  {
    size_t num_dets = hal_detect_box_list_len(boxes);
    app->totalFrames++;

    // Frame limit — stop after N frames and print timing
    if (app->maxFrames > 0 && app->totalFrames >= app->maxFrames) {
      if (num_dets > 0) {
        app->framesWithDetections++;
        app->totalDetections += num_dets;
      }
      gettimeofday(&endTime, NULL);
      app->postproc.record(timeDiffMs(startTime, endTime));
      if (boxes) hal_detect_box_list_free(boxes);
      for (int j = 0; j < NUM_OUTPUTS; j++) {
        if (hal_outputs[j]) hal_tensor_free(hal_outputs[j]);
      }
      for (int j = 0; j < NUM_OUTPUTS; j++) {
        if (out_mapped[j])
          gst_memory_unmap(gst_buffer_peek_memory(buffer, j), &out_maps[j]);
      }
      printTiming(app);
      g_main_loop_quit(app->loop);
      return;
    }

    if (num_dets > 0) {
      app->framesWithDetections++;
      app->totalDetections += num_dets;

      // Print first frame detections
      static bool first_print = true;
      if (first_print) {
        g_print("Detected objects (%zu):\n", num_dets);
        for (size_t d = 0; d < num_dets; d++) {
          hal_detect_box box;
          if (hal_detect_box_list_get(boxes, d, &box) == 0) {
            /* HAL decoder with normalized:false returns model pixel coords
             * directly — do NOT multiply by MODEL_INPUT_SIZE. */
            float px = (box.xmin - app->letterbox.padX) / app->letterbox.scale;
            float py = (box.ymin - app->letterbox.padY) / app->letterbox.scale;
            float bw = (box.xmax - box.xmin) / app->letterbox.scale;
            float bh = (box.ymax - box.ymin) / app->letterbox.scale;

            const char *name = (box.label < NUM_CLASSES) ? cocoClassNames[box.label] : "unknown";
            g_print("  - %s (%.2f) at (%.0f,%.0f) %.0fx%.0f\n",
                    name, box.score, px, py, bw, bh);
          }
        }
        g_print("\n");
        first_print = false;
      }
    }
  }

cleanup:
  if (boxes)
    hal_detect_box_list_free(boxes);
  for (int j = 0; j < NUM_OUTPUTS; j++) {
    if (hal_outputs[j])
      hal_tensor_free(hal_outputs[j]);
  }
  for (int j = 0; j < NUM_OUTPUTS; j++) {
    if (out_mapped[j])
      gst_memory_unmap(gst_buffer_peek_memory(buffer, j), &out_maps[j]);
  }

  gettimeofday(&endTime, NULL);
  app->postproc.record(timeDiffMs(startTime, endTime));
}


/* --- SIGINT handler ------------------------------------------------------ */

static gboolean sigintHandler(gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  g_print("\nSIGINT — stopping.\n");
  printTiming(app);
  g_main_loop_quit(app->loop);
  return TRUE;
}


/* --- Command line -------------------------------------------------------- */

static int parseArgs(int argc, char **argv, std::string &model,
                     std::string &camera, std::string &video,
                     std::string &image, std::string &config,
                     int &numFrames)
{
  static struct option long_opts[] = {
    {"help",   no_argument,       0, 'h'},
    {"model",  required_argument, 0, 'm'},
    {"camera", required_argument, 0, 'd'},
    {"video",  required_argument, 0, 'v'},
    {"image",  required_argument, 0, 'i'},
    {"config", required_argument, 0, 'c'},
    {"frames", required_argument, 0, 'n'},
    {0, 0, 0, 0}
  };

  int c;
  while ((c = getopt_long(argc, argv, "hm:d:v:i:c:n:", long_opts, NULL)) != -1) {
    switch (c) {
      case 'h':
        std::cout
            << "YOLOv8n 640x640 for Kinara Ara-2 NPU — EdgeFirst CameraAdaptor + HAL\n\n"
            << "Usage: " << argv[0] << " -m MODEL.dvm [-v VIDEO | -i IMAGE | -d CAMERA] [-c config.yaml] [-n FRAMES]\n\n"
            << "Options:\n"
            << "  -m, --model PATH    Path to YOLOv8n .dvm model [required]\n"
            << "  -d, --camera DEVICE Camera device (default: " << DEFAULT_CAMERA_DEVICE << ")\n"
            << "  -v, --video PATH    H.264 MP4 video input\n"
            << "  -i, --image PATH    Static JPEG image input\n"
            << "  -c, --config PATH   EdgeFirst YAML config (auto-detects edgefirst.yaml next to model)\n"
            << "  -n, --frames N      Stop after N frames and print timing (0=infinite)\n"
            << "  -h, --help          Show this help\n\n"
            << "EdgeFirst optimizations:\n"
            << "  * edgefirstcameraadaptor: fused NV12->int8 CHW preprocessing (HAL)\n"
            << "  * Ara-2 NPU: native int8 CHW inference (no internal preprocessing)\n"
            << "  * EdgeFirst HAL: quantized NMS (no CPU dequantization)\n";
        return 1;
      case 'm': model = optarg; break;
      case 'd': camera = optarg; break;
      case 'v': video = optarg; break;
      case 'i': image = optarg; break;
      case 'c': config = optarg; break;
      case 'n': numFrames = atoi(optarg); break;
    }
  }
  return 0;
}


/* --- Main ---------------------------------------------------------------- */

int main(int argc, char **argv)
{
  std::string model, camera = DEFAULT_CAMERA_DEVICE, video, image, config;
  int numFrames = 0;
  if (parseArgs(argc, argv, model, camera, video, image, config, numFrames))
    return 0;

  if (model.empty()) {
    g_printerr("ERROR: provide model path with -m\n");
    return 1;
  }

  gst_init(&argc, &argv);

  // Auto-detect edgefirst.yaml next to model when -c not provided
  std::string config_path = config;
  if (config_path.empty()) {
    std::string model_dir = model.substr(0, model.find_last_of('/'));
    if (model_dir == model)
      model_dir = ".";
    config_path = model_dir + "/edgefirst.yaml";
    if (access(config_path.c_str(), R_OK) != 0) {
      g_printerr("ERROR: no config file specified and edgefirst.yaml not found "
                  "next to model.\nProvide -c /path/to/edgefirst.yaml\n");
      return 1;
    }
    g_print("Auto-detected config: %s\n", config_path.c_str());
  }

  // Initialize HAL decoder
  hal_decoder_params params = hal_decoder_params_default();
  params.config_file = config_path.c_str();
  params.score_threshold = CONF_THRESHOLD;
  params.iou_threshold = NMS_IOU_THRESHOLD;
  params.nms = HAL_NMS_CLASS_AGNOSTIC;

  hal_decoder *decoder = hal_decoder_new(&params);
  if (!decoder) {
    g_printerr("ERROR: failed to create HAL decoder: %s\n", strerror(errno));
    return 1;
  }

  char *model_type = hal_decoder_model_type(decoder);
  g_print("HAL decoder: type=%s\n", model_type ? model_type : "unknown");
  free(model_type);

  // Letterbox params will be queried from cameraadaptor after caps negotiation

  // Build source element
  std::string sourceStr;
  if (!image.empty()) {
    char *s = g_strdup_printf(
        "filesrc location=%s ! jpegdec ! imxvideoconvert_g2d ! "
        "video/x-raw,width=%d,height=%d ! imagefreeze",
        image.c_str(), SOURCE_WIDTH, SOURCE_HEIGHT);
    sourceStr = s;
    g_free(s);
    g_print("Input: image (%s)\n", image.c_str());
  } else if (!video.empty()) {
    char *s = g_strdup_printf(
        "filesrc location=%s ! qtdemux ! h264parse ! v4l2h264dec",
        video.c_str());
    sourceStr = s;
    g_free(s);
    g_print("Input: video (%s)\n", video.c_str());
  } else {
    char *s = g_strdup_printf(
        "v4l2src device=%s ! video/x-raw,format=NV12,width=%d,height=%d,framerate=30/1",
        camera.c_str(), SOURCE_WIDTH, SOURCE_HEIGHT);
    sourceStr = s;
    g_free(s);
    g_print("Input: camera (%s)\n", camera.c_str());
  }

  // Pipeline: source → cameraadaptor → ara2 → tensor_sink
  gchar *pipelineStr = g_strdup_printf(
      "%s ! "
      "queue name=thread-nn leaky=2 max-size-buffers=2 ! "
      "edgefirstcameraadaptor name=preproc model-width=%d model-height=%d "
      "model-dtype=int8 model-layout=chw letterbox=true ! "
      "tensor_filter name=tfilter framework=ara2 model=%s "
      "custom=EnableStats:true latency=1 ! "
      "tensor_sink name=inferenceOutput",
      sourceStr.c_str(),
      MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
      model.c_str());

  g_print("Pipeline:\n%s\n\n", pipelineStr);

  if (numFrames > 0)
    g_print("Frames: %d\n", numFrames);

  AppData app = {};
  app.decoder = decoder;
  app.maxFrames = numFrames;
  app.preproc.reset();
  app.inference.reset();
  app.postproc.reset();
  app.e2e.reset();
  app.firstFrame = true;

  app.loop = g_main_loop_new(NULL, FALSE);
  app.pipeline = gst_parse_launch(pipelineStr, NULL);
  g_free(pipelineStr);

  if (!app.pipeline) {
    g_printerr("ERROR: failed to create pipeline\n");
    hal_decoder_free(decoder);
    return 1;
  }

  // Bus
  app.bus = gst_element_get_bus(app.pipeline);
  gst_bus_add_signal_watch(app.bus);
  g_signal_connect(app.bus, "message", G_CALLBACK(busCallback), &app);

  // tensor_sink
  GstElement *tsink = gst_bin_get_by_name(GST_BIN(app.pipeline), "inferenceOutput");
  g_signal_connect(tsink, "new-data", G_CALLBACK(newDataCallback), &app);
  gst_object_unref(tsink);

  // tensor_filter for latency
  app.tensorFilter = gst_bin_get_by_name(GST_BIN(app.pipeline), "tfilter");

  // Preprocessing timing probes (cameraadaptor sink → src)
  app.cameraadaptor = gst_bin_get_by_name(GST_BIN(app.pipeline), "preproc");
  if (app.cameraadaptor) {
    GstPad *sinkPad = gst_element_get_static_pad(app.cameraadaptor, "sink");
    GstPad *srcPad = gst_element_get_static_pad(app.cameraadaptor, "src");
    if (sinkPad) {
      gst_pad_add_probe(sinkPad, GST_PAD_PROBE_TYPE_BUFFER, preprocStartProbe, &app, NULL);
      gst_object_unref(sinkPad);
    }
    if (srcPad) {
      gst_pad_add_probe(srcPad, GST_PAD_PROBE_TYPE_BUFFER, preprocEndProbe, &app, NULL);
      gst_object_unref(srcPad);
    }
  }

  g_unix_signal_add(SIGINT, sigintHandler, &app);

  // Run — letterbox params are queried from cameraadaptor on first frame
  gst_element_set_state(app.pipeline, GST_STATE_PLAYING);
  g_main_loop_run(app.loop);

  // Cleanup
  gst_element_set_state(app.pipeline, GST_STATE_NULL);
  if (app.tensorFilter)
    gst_object_unref(app.tensorFilter);
  if (app.cameraadaptor)
    gst_object_unref(app.cameraadaptor);
  gst_object_unref(app.bus);
  gst_object_unref(app.pipeline);
  g_main_loop_unref(app.loop);
  hal_decoder_free(decoder);

  return 0;
}
