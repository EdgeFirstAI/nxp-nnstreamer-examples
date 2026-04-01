/*
 * YOLOv8n Instance Segmentation — EdgeFirst Overlay Pipeline
 * Copyright (C) 2026 Au-Zone Technologies
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Demonstrates YOLOv8n instance segmentation using the EdgeFirst native
 * GStreamer pipeline: edgefirstcameraadaptor for preprocessing and
 * edgefirstoverlay for GPU-accelerated mask/box drawing with detection
 * callbacks.
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <string>
#include <map>
#include <sys/time.h>

#include "common/yolov8_common.hpp"
#include <gst/edgefirst/edgefirstdetection.h>

#define DEFAULT_MODEL   "/opt/edgefirst/models/yolov8n-seg_640x640.tflite"
#define DEFAULT_CAMERA  "/dev/video3"


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
};


/* ---- Timing report ---------------------------------------------------- */

static void printTimingReport(void *userData)
{
  AppData *app = (AppData *)userData;
  if (app->timingPrinted) return;
  app->timingPrinted = true;

  printf("\n=== YOLOv8n-seg Timing Report ===\n");

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

  printf("=================================\n");
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

  app->frameCount++;

  guint nBoxes = boxes ? edgefirst_detect_box_list_get_length(boxes) : 0;
  guint nSegs  = segs  ? edgefirst_segmentation_list_get_length(segs) : 0;

  if (nBoxes > 0) {
    app->totalDetections += nBoxes;
    app->framesWithDetections++;
  }

  /* Print per-frame detections if -D is active */
  if (app->detections && nBoxes > 0) {
    printf("[frame %d] %u detections, %u masks\n",
           app->frameCount, nBoxes, nSegs);

    for (guint i = 0; i < nBoxes; i++) {
      EdgeFirstDetectBox *box = edgefirst_detect_box_list_get(boxes, i);
      if (!box) continue;

      const char *name = (box->class_id >= 0 && box->class_id < NUM_CLASSES)
          ? cocoClassNames[box->class_id] : "?";

      /* Mask dimensions if available */
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

  /* Frame count limit: send EOS when reached */
  if (app->numFrames > 0 && app->frameCount >= app->numFrames)
    g_main_loop_quit(app->loop);
}


/* ---- Main ------------------------------------------------------------- */

int main(int argc, char **argv)
{
  ParsedArgs pargs;
  pargs.model  = DEFAULT_MODEL;
  pargs.camera = DEFAULT_CAMERA;

  uint32_t flags = ARG_MODEL | ARG_CAMERA | ARG_VIDEO |
                   ARG_HEADLESS | ARG_INSTRUMENTED | ARG_NUM_FRAMES |
                   ARG_COMPUTE | ARG_DETECTIONS;

  int ret = parseArgs(argc, argv, flags,
      "YOLOv8n Instance Segmentation — EdgeFirst Overlay Pipeline", pargs);
  if (ret != 0) return ret > 0 ? 0 : 1;

  gst_init(&argc, &argv);

  printf("YOLOv8n-seg — EdgeFirst Overlay Pipeline\n");
  printf("  Model:   %s\n", pargs.model.c_str());
  if (!pargs.video.empty())
    printf("  Input:   video (%s)\n", pargs.video.c_str());
  else
    printf("  Input:   camera (%s)\n", pargs.camera.c_str());
  printf("  Mode:    %s\n", pargs.headless ? "headless" : "display");
  if (pargs.numFrames > 0)
    printf("  Frames:  %d\n", pargs.numFrames);
  if (!pargs.compute.empty())
    printf("  Compute: %s\n", pargs.compute.c_str());
  printf("\n");

  /* ---- Build pipeline via gst_parse_launch ---- */

  GMainLoop *loop = g_main_loop_new(NULL, FALSE);

  /* Build source element string */
  InputSource inputSrc = determineInputSource(pargs, false);
  char *srcStr = buildSourceElement(inputSrc, pargs);

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

  /* Full pipeline string — gst_parse_launch handles dynamic pads (qtdemux) */
  gchar *pipeStr = g_strdup_printf(
      "%s ! tee name=t "
      "t. ! queue name=q-disp leaky=2 max-size-buffers=2 "
      "   ! edgefirstoverlay name=ov score-threshold=0.25 iou-threshold=0.45 "
      "     decoder-version=yolov8%s "
      "   ! %s "
      "t. ! queue name=q-nn leaky=2 max-size-buffers=2 "
      "   ! edgefirstcameraadaptor name=ca model-width=640 model-height=640 "
      "     model-dtype=uint8 model-colorspace=rgba letterbox=true%s "
      "   ! tensor_filter name=tfilter framework=tensorflow2-lite "
      "     model=%s "
      "     custom=Delegate:External,ExtDelegateLib:libvx_delegate.so,CameraAdaptor:rgba,DmaBuf:true "
      "     latency=1 "
      "   ! ov.tensors",
      srcStr, computeOV.c_str(), sinkStr, computeCA.c_str(),
      pargs.model.c_str());
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

  /* ---- Run ---- */

  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);

  /* ---- Cleanup ---- */

  gst_element_set_state(pipeline, GST_STATE_NULL);

  if (app.instrumented)
    printTimingReport(&app);

  gst_object_unref(overlay);
  gst_object_unref(ca);
  gst_object_unref(tfilter);
  gst_object_unref(app.bus);
  gst_object_unref(pipeline);
  g_main_loop_unref(loop);
  app.ptsTracker.destroy();

  return 0;
}
