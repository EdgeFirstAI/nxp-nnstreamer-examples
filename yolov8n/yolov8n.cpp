/*
 * YOLOv8n — EdgeFirst Overlay Pipeline (Unified Detection + Segmentation)
 * Copyright (C) 2026 Au-Zone Technologies
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Single binary supporting YOLOv8n detection and instance segmentation
 * across all EdgeFirst-supported backends and platforms:
 *
 *   Backends:  TFLite (VX Delegate / Neutron) and Kinara Ara-2
 *   Platforms: i.MX 8M Plus and i.MX 95
 *
 * The model type (detection vs. segmentation) is auto-detected by the
 * edgefirstoverlay element from the tensor shapes — no flag needed.
 * Backend is auto-detected from the model file extension (.tflite / .dvm).
 * Platform must be specified with -p.
 *
 * Pipeline:
 *   [source] -> tee name=t
 *     t. -> queue -> edgefirstoverlay name=ov -> [sink]
 *     t. -> queue -> edgefirstcameraadaptor -> tensor_filter -> ov.tensors
 *
 * The overlay element handles all post-processing internally: HAL decoder
 * init, tensor mapping, NMS, mask materialization, and GPU rendering.
 * The application just builds the pipeline, optionally connects to the
 * new-detection signal for printing, and runs the main loop.
 */

#include <gst/gst.h>
#include <gst/allocators/gstdmabuf.h>
#include <glib-unix.h>
#include <getopt.h>
#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <sys/time.h>

#include "common/yolov8_common.hpp"
#include <gst/edgefirst/edgefirstdetection.h>


/* ---- Backend / platform configuration --------------------------------- */

enum Backend  { BACKEND_TFLITE_VX, BACKEND_TFLITE_NEUTRON, BACKEND_ARA2 };
enum Platform { PLATFORM_IMX8MP, PLATFORM_IMX95 };

struct PlatformConfig {
  const char *name;
  const char *defaultCamera;
  bool usesLibcamerasrc;
};

static const PlatformConfig platformConfigs[] = {
  [PLATFORM_IMX8MP] = {
    .name = "i.MX 8M Plus",
    .defaultCamera = "/dev/video3",
    .usesLibcamerasrc = false,
  },
  [PLATFORM_IMX95] = {
    .name = "i.MX 95",
    .defaultCamera = NULL,
    .usesLibcamerasrc = true,
  },
};

static bool hasSuffix(const std::string &str, const char *suffix) {
  std::string s = str, sfx = suffix;
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  std::transform(sfx.begin(), sfx.end(), sfx.begin(), ::tolower);
  return s.size() >= sfx.size() && s.compare(s.size() - sfx.size(), sfx.size(), sfx) == 0;
}

static const char *backendName(Backend b) {
  switch (b) {
    case BACKEND_TFLITE_VX:      return "TFLite VX Delegate";
    case BACKEND_TFLITE_NEUTRON: return "TFLite Neutron Delegate";
    case BACKEND_ARA2:           return "Ara-2";
  }
  return "Unknown";
}


/* ---- Application data ------------------------------------------------- */

struct AppData {
  GstElement  *pipeline;
  GMainLoop   *loop;
  GstBus      *bus;
  GstElement  *tensorFilter;

  /* Timing */
  TimingMetric  preproc;       /* cameraadaptor sink -> src */
  TimingMetric  inference;
  TimingMetric  e2e;           /* cameraadaptor sink -> overlay src */
  ThroughputTracker throughput;
  PtsTracker    ptsTracker;

  struct timeval preprocStart;

  /* Detection stats */
  int  totalDetections;
  int  framesWithDetections;
  int  frameCount;

  /* Config */
  bool headless;
  bool instrumented;
  bool detections;
  int  numFrames;

  /* Infrastructure */
  BusCallbackCtx busCtx;
  bool timingPrinted;

  /* Per-element + full pipeline latency (shared helper) */
  PipelineProbes probes;
};


/* ---- Timing report ---------------------------------------------------- */

static void printTimingReport(void *userData)
{
  AppData *app = (AppData *)userData;
  if (app->timingPrinted) return;
  app->timingPrinted = true;

  printf("\n=== YOLOv8n Timing Report ===\n");

  int frames = app->throughput.metric.count > 0
      ? app->throughput.metric.count : app->frameCount;
  printf("  Frames processed:  %d\n", frames);
  printf("  Total detections:  %d (in %d frames)\n",
         app->totalDetections, app->framesWithDetections);

  printMetric("Preprocess:", "edgefirstcameraadaptor sink->src", app->preproc);
  printMetric("Inference:", "tensor_filter latency", app->inference);
  printMetric("End-to-End:", "cameraadaptor sink -> overlay src", app->e2e);

  if (app->throughput.metric.count > 0) {
    double avgMs = app->throughput.metric.avg();
    printf("  Throughput:  %.1f FPS\n", avgMs > 0 ? 1000.0 / avgMs : 0);
  }

  printf("=============================\n");
}


/* ---- Pad probes for timing -------------------------------------------- */

static GstPadProbeReturn caSinkProbe(GstPad *, GstPadProbeInfo *info,
                                     gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  gettimeofday(&app->preprocStart, NULL);

  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  app->ptsTracker.recordStart(buffer, app->preprocStart);
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn caSrcProbe(GstPad *, GstPadProbeInfo *,
                                    gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  struct timeval now;
  gettimeofday(&now, NULL);
  app->preproc.record(timeDiffMs(app->preprocStart, now));
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn overlaySrcProbe(GstPad *, GstPadProbeInfo *info,
                                         gpointer user_data)
{
  AppData *app = (AppData *)user_data;
  struct timeval now;
  gettimeofday(&now, NULL);

  /* E2E: cameraadaptor sink -> overlay src (PTS-correlated) */
  GstBuffer *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  struct timeval ptsStart;
  if (app->ptsTracker.consumeStart(buffer, ptsStart))
    app->e2e.record(timeDiffMs(ptsStart, now));

  return GST_PAD_PROBE_OK;
}


/* ---- Detection callback ----------------------------------------------- */

static void onNewDetection(GstElement *,
                           EdgeFirstDetectBoxList *boxes,
                           EdgeFirstSegmentationList *segs,
                           gpointer user_data)
{
  AppData *app = (AppData *)user_data;

  struct timeval now;
  gettimeofday(&now, NULL);
  app->throughput.tick(now);

  /* Query inference latency each frame */
  queryInferenceLatency(app->tensorFilter, app->inference);
  app->probes.recordInference();

  app->frameCount++;

  guint nBoxes = boxes ? edgefirst_detect_box_list_get_length(boxes) : 0;
  guint nSegs  = segs  ? edgefirst_segmentation_list_get_length(segs) : 0;

  if (nBoxes > 0) {
    app->totalDetections += nBoxes;
    app->framesWithDetections++;
  }

  /* Print per-frame detections if -D is active */
  if (app->detections && nBoxes > 0) {
    printf("[frame %d] %u detections", app->frameCount, nBoxes);
    if (nSegs > 0)
      printf(", %u masks", nSegs);
    printf("\n");

    for (guint i = 0; i < nBoxes; i++) {
      EdgeFirstDetectBox *box = edgefirst_detect_box_list_get(boxes, i);
      if (!box) continue;

      const char *name = (box->class_id >= 0 && box->class_id < NUM_CLASSES)
          ? cocoClassNames[box->class_id] : "?";

      if (segs && i < nSegs) {
        EdgeFirstSegmentation *seg = edgefirst_segmentation_list_get(segs, i);
        if (seg) {
          printf("  [%u] %-14s %3.0f%%  box=(%.2f,%.2f,%.2f,%.2f)  mask=%ux%u\n",
                 i, name, box->score * 100.0f,
                 box->x1, box->y1, box->x2, box->y2,
                 seg->width, seg->height);
          edgefirst_segmentation_free(seg);
        } else {
          printf("  [%u] %-14s %3.0f%%  box=(%.2f,%.2f,%.2f,%.2f)\n",
                 i, name, box->score * 100.0f,
                 box->x1, box->y1, box->x2, box->y2);
        }
      } else {
        printf("  [%u] %-14s %3.0f%%  box=(%.2f,%.2f,%.2f,%.2f)\n",
               i, name, box->score * 100.0f,
               box->x1, box->y1, box->x2, box->y2);
      }

      edgefirst_detect_box_free(box);
    }
  }

  /* Shared full-latency tracking */
  struct timeval cbEnd;
  gettimeofday(&cbEnd, NULL);
  app->probes.recordPost(now, cbEnd);

  /* Frame count limit: send EOS when reached */
  if (app->numFrames > 0 && app->frameCount >= app->numFrames)
    g_main_loop_quit(app->loop);
}


/* ---- Main ------------------------------------------------------------- */

int main(int argc, char **argv)
{
  ParsedArgs pargs;
  pargs.model  = "";  /* required — no default */
  pargs.camera = "";

  uint32_t flags = ARG_MODEL | ARG_CAMERA | ARG_VIDEO |
                   ARG_HEADLESS | ARG_INSTRUMENTED | ARG_NUM_FRAMES |
                   ARG_COMPUTE | ARG_DETECTIONS | ARG_PLATFORM;

  int ret = parseArgs(argc, argv, flags,
      "YOLOv8n — EdgeFirst Overlay Pipeline", pargs);
  if (ret != 0) return ret > 0 ? 0 : 1;

  /* ---- Validate model ---- */
  if (pargs.model.empty()) {
    fprintf(stderr, "ERROR: Model file is required. Use -m <path>.\n"
                    "  Detection:     -m yolov8n_640x640.tflite\n"
                    "  Segmentation:  -m yolov8n-seg_640x640.tflite\n"
                    "  Ara-2 (DVM):   -m yolov8n_640x640.dvm\n");
    return 1;
  }

  /* ---- Platform selection ---- */
  if (pargs.platformStr.empty()) {
    fprintf(stderr, "ERROR: Platform is required. Use -p imx8mp or -p imx95.\n");
    return 1;
  }

  Platform platform;
  if (pargs.platformStr == "imx95")
    platform = PLATFORM_IMX95;
  else if (pargs.platformStr == "imx8mp")
    platform = PLATFORM_IMX8MP;
  else {
    fprintf(stderr, "ERROR: Unknown platform '%s'. Use imx8mp or imx95.\n",
            pargs.platformStr.c_str());
    return 1;
  }

  /* ---- Detect backend from model extension ---- */
  Backend backend;
  if (hasSuffix(pargs.model, ".dvm")) {
    backend = BACKEND_ARA2;
  } else if (platform == PLATFORM_IMX95) {
    backend = BACKEND_TFLITE_NEUTRON;
  } else {
    backend = BACKEND_TFLITE_VX;
  }

  const PlatformConfig &plat = platformConfigs[platform];
  if (pargs.camera.empty() && plat.defaultCamera)
    pargs.camera = plat.defaultCamera;
  if (platform == PLATFORM_IMX95)
    setupImx95Environment(false);

  gst_init(&argc, &argv);

  printf("YOLOv8n — EdgeFirst Overlay Pipeline\n");
  printf("  Platform: %s\n", plat.name);
  printf("  Backend:  %s\n", backendName(backend));
  printf("  Model:    %s\n", pargs.model.c_str());
  if (!pargs.video.empty())
    printf("  Input:    video (%s)\n", pargs.video.c_str());
  else
    printf("  Input:    camera (%s)\n", pargs.camera.c_str());
  printf("  Mode:     %s\n", pargs.headless ? "headless" : "display");
  if (pargs.numFrames > 0)
    printf("  Frames:   %d\n", pargs.numFrames);
  if (!pargs.compute.empty())
    printf("  Compute:  %s\n", pargs.compute.c_str());
  printf("\n");

  /* ---- Build pipeline via gst_parse_launch ---- */

  GMainLoop *loop = g_main_loop_new(NULL, FALSE);

  /* Build source element string */
  InputSource inputSrc = determineInputSource(pargs, plat.usesLibcamerasrc);
  char *srcStr = buildSourceElement(inputSrc, pargs);

  /* i.MX 8M Plus NV12 workaround: v4l2h264dec outputs NV12 and the Vivante
   * GPU NV12 sampling path is unreliable.  Insert HW G2D colour-space
   * conversion to RGBA for video sources.  YUYV camera input (v4l2src) is
   * already fast on Vivante.  imx95 (Mali GPU) handles NV12 natively. */
  if (platform == PLATFORM_IMX8MP && inputSrc == INPUT_VIDEO) {
    char *orig = srcStr;
    srcStr = g_strdup_printf(
        "%s ! imxvideoconvert_g2d ! video/x-raw,format=RGBA,width=%d,height=%d",
        orig, SOURCE_WIDTH, SOURCE_HEIGHT);
    g_free(orig);
    printf("  NV12 workaround: inserting imxvideoconvert_g2d → RGBA before tee\n");
  }

  /* Pace file-based sources to their natural PTS rate */
  if (inputSrc == INPUT_VIDEO) {
    char *orig = srcStr;
    srcStr = g_strdup_printf("%s ! identity sync=true", orig);
    g_free(orig);
  }

  /* Build compute property strings */
  std::string computeCA, computeOV;
  if (!pargs.compute.empty()) {
    computeCA = std::string(" compute=") + pargs.compute;
    computeOV = std::string(" compute=") + pargs.compute;
  }

  /* Sink element */
  const char *sinkStr = pargs.headless
      ? "fakesink name=sink sync=false"
      : "waylandsink name=sink";

  /* Backend-specific cameraadaptor and tensor_filter properties */
  const char *caDtype, *caLayout, *tfFramework, *tfCustom;
  const char *ovNormalized;
  switch (backend) {
    case BACKEND_ARA2:
      caDtype      = "int8";
      caLayout     = "model-layout=chw";
      tfFramework  = "ara2";
      tfCustom     = "custom=EnableStats:true";
      ovNormalized = " normalized=false";
      break;
    case BACKEND_TFLITE_NEUTRON:
      caDtype      = "uint8";
      caLayout     = "model-colorspace=rgba";
      tfFramework  = "tensorflow2-lite";
      tfCustom     = "custom=Delegate:External,ExtDelegateLib:libneutron_delegate.so";
      ovNormalized = "";
      break;
    case BACKEND_TFLITE_VX:
    default:
      caDtype      = "uint8";
      caLayout     = "model-colorspace=rgba";
      tfFramework  = "tensorflow2-lite";
      tfCustom     = "custom=Delegate:External,ExtDelegateLib:libvx_delegate.so,CameraAdaptor:rgba,DmaBuf:true";
      ovNormalized = "";
      break;
  }

  /* Full pipeline string — gst_parse_launch handles dynamic pads (qtdemux) */
  gchar *pipeStr = g_strdup_printf(
      "%s ! tee name=t "
      "t. ! queue name=q-disp leaky=2 max-size-buffers=2 "
      "   ! edgefirstoverlay name=ov score-threshold=0.25 iou-threshold=0.45 "
      "     decoder-version=yolov8%s%s "
      "   ! %s "
      "t. ! queue name=q-nn leaky=2 max-size-buffers=2 "
      "   ! edgefirstcameraadaptor name=ca model-width=640 model-height=640 "
      "     model-dtype=%s %s letterbox=true%s "
      "   ! tensor_filter name=tfilter framework=%s "
      "     model=%s "
      "     %s "
      "     latency=1 "
      "   ! ov.tensors",
      srcStr, ovNormalized, computeOV.c_str(), sinkStr,
      caDtype, caLayout, computeCA.c_str(),
      tfFramework, pargs.model.c_str(), tfCustom);
  g_free(srcStr);

  GError *err = NULL;
  GstElement *pipeline = gst_parse_launch(pipeStr, &err);
  g_free(pipeStr);
  if (!pipeline) {
    g_printerr("Failed to create pipeline: %s\n", err ? err->message : "unknown");
    if (err) g_error_free(err);
    return 1;
  }

  /* Look up named elements for probes and signals */
  GstElement *overlay = gst_bin_get_by_name(GST_BIN(pipeline), "ov");
  GstElement *ca      = gst_bin_get_by_name(GST_BIN(pipeline), "ca");
  GstElement *tfilter = gst_bin_get_by_name(GST_BIN(pipeline), "tfilter");
  if (!overlay || !ca || !tfilter) {
    g_printerr("Failed to find named elements in pipeline\n");
    return 1;
  }

  /* ---- Initialize app data ---- */

  AppData app = {};
  app.pipeline     = pipeline;
  app.loop         = loop;
  app.tensorFilter = tfilter;
  app.headless     = pargs.headless;
  app.instrumented = pargs.instrumented;
  app.detections   = pargs.detections;
  app.numFrames    = pargs.numFrames;
  app.preproc.reset();
  app.inference.reset();
  app.e2e.reset();
  app.throughput.reset();
  app.ptsTracker.init();
  app.probes.reset();

  bool startedOnce = false;
  app.busCtx.pipeline    = pipeline;
  app.busCtx.loop        = loop;
  app.busCtx.playing     = NULL;
  app.busCtx.startedOnce = &startedOnce;
  app.busCtx.videoLoop   = !pargs.video.empty();
  app.busCtx.videoRate   = 1.0;
  app.busCtx.printTiming = app.instrumented ? printTimingReport : NULL;
  app.busCtx.appData     = &app;

  /* ---- Connect bus + signals ---- */

  app.bus = gst_element_get_bus(pipeline);
  gst_bus_add_signal_watch(app.bus);
  g_signal_connect(app.bus, "message", G_CALLBACK(commonBusCallback), &app.busCtx);

  /* Detection callback */
  g_signal_connect(overlay, "new-detection",
                   G_CALLBACK(onNewDetection), &app);

  /* SIGINT handler */
  g_unix_signal_add(SIGINT, commonSigintHandler, &app.busCtx);

  /* ---- Install pad probes for timing ---- */

  if (app.instrumented) {
    /* Preprocess: cameraadaptor sink -> src */
    GstPad *caSink = gst_element_get_static_pad(ca, "sink");
    GstPad *caSrc  = gst_element_get_static_pad(ca, "src");
    if (caSink) {
      gst_pad_add_probe(caSink, GST_PAD_PROBE_TYPE_BUFFER,
                         caSinkProbe, &app, NULL);
      gst_object_unref(caSink);
    }
    if (caSrc) {
      gst_pad_add_probe(caSrc, GST_PAD_PROBE_TYPE_BUFFER,
                         caSrcProbe, &app, NULL);
      gst_object_unref(caSrc);
    }

    /* E2E: overlay src pad */
    GstPad *ovSrc = gst_element_get_static_pad(overlay, "src");
    if (ovSrc) {
      gst_pad_add_probe(ovSrc, GST_PAD_PROBE_TYPE_BUFFER,
                         overlaySrcProbe, &app, NULL);
      gst_object_unref(ovSrc);
    }
  }

  /* Shared per-element + full pipeline latency probes */
  app.probes.install(pipeline, "q-nn", "ca", "tfilter");

  /* ---- Run ---- */

  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);

  /* ---- Cleanup ---- */

  gst_element_set_state(pipeline, GST_STATE_NULL);

  if (app.instrumented)
    printTimingReport(&app);
  app.probes.printReport("YOLOv8n — EdgeFirst Overlay Pipeline (per-element probes)");
  app.probes.teardown();

  gst_object_unref(overlay);
  gst_object_unref(ca);
  gst_object_unref(tfilter);
  gst_object_unref(app.bus);
  gst_object_unref(pipeline);
  g_main_loop_unref(loop);
  app.ptsTracker.destroy();

  return 0;
}
