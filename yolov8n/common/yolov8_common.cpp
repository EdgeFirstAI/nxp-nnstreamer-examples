/**
 * Copyright 2025 NXP
 * Copyright 2026 EdgeFirst AI (Au-Zone Technologies)
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Shared implementation for YOLOv8n 640x640 demos.
 */

#include "yolov8_common.hpp"

#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <vector>



/* ─── COCO class names ────────────────────────────────────────────── */

const char *cocoClassNames[NUM_CLASSES] = {
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
  "truck", "boat", "traffic light", "fire hydrant", "stop sign",
  "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
  "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
  "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
  "couch", "potted plant", "bed", "dining table", "toilet", "tv",
  "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
  "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
  "scissors", "teddy bear", "hair drier", "toothbrush"
};


std::vector<std::string> getCocoClassNames() {
  return std::vector<std::string>(cocoClassNames, cocoClassNames + NUM_CLASSES);
}


/* ─── Letterbox ───────────────────────────────────────────────────── */

LetterboxParams calculateLetterbox(int srcWidth, int srcHeight) {
  float scaleX = (float)MODEL_INPUT_SIZE / srcWidth;
  float scaleY = (float)MODEL_INPUT_SIZE / srcHeight;
  float scale = std::min(scaleX, scaleY);
  int scaledW = (int)(srcWidth * scale);
  int scaledH = (int)(srcHeight * scale);
  int padLeft = (MODEL_INPUT_SIZE - scaledW) / 2;
  int padTop = (MODEL_INPUT_SIZE - scaledH) / 2;
  int padRight = MODEL_INPUT_SIZE - scaledW - padLeft;
  int padBottom = MODEL_INPUT_SIZE - scaledH - padTop;
  return {scale, padLeft, padTop, scaledW, scaledH, padRight, padBottom};
}


/* ─── Quantization meta extraction ────────────────────────────────── */

bool extractQuantParams(GstBuffer *buffer, unsigned int tensorIdx, QuantParams &out) {
  GstNnsTensorQuantMeta *qm = gst_buffer_get_nns_tensor_quant_meta(buffer);
  if (!qm || tensorIdx >= qm->num_tensors)
    return false;
  const NnsTensorQuantInfo *qi = &qm->quant[tensorIdx];
  if (qi->scheme == NNS_QUANT_NONE || qi->num_params == 0)
    return false;
  out.scale = qi->scales[0];
  out.zeroPoint = qi->zero_points[0];
  return true;
}


/* ─── Print metric ────────────────────────────────────────────────── */

void printMetric(const char *label, const char *desc, const TimingMetric &metric) {
  if (metric.count > 0) {
    printf("  %s\n", label);
    if (desc)
      printf("     (%s)\n", desc);
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d frames]\n",
           metric.avg(), metric.minMs, metric.maxMs, metric.count);
  }
}


/* ─── Argument Parsing ────────────────────────────────────────────── */

int parseArgs(int argc, char **argv, uint32_t supportedFlags,
              const char *binaryDesc, ParsedArgs &args)
{
  // Build getopt_long options dynamically from supportedFlags.
  // Fixed options: -h/--help is always present.
  std::vector<struct option> opts;
  std::string shortOpts = "h";

  opts.push_back({"help", no_argument, 0, 'h'});

  if (supportedFlags & ARG_MODEL) {
    opts.push_back({"model", required_argument, 0, 'm'});
    shortOpts += "m:";
  }
  if (supportedFlags & ARG_CAMERA) {
    opts.push_back({"camera", required_argument, 0, 'c'});
    shortOpts += "c:";
  }
  if (supportedFlags & ARG_VIDEO) {
    opts.push_back({"video", required_argument, 0, 'v'});
    shortOpts += "v:";
  }
  if (supportedFlags & ARG_IMAGE) {
    opts.push_back({"image", required_argument, 0, 'i'});
    shortOpts += "i:";
  }
  if (supportedFlags & ARG_HEADLESS) {
    opts.push_back({"headless", no_argument, 0, 'H'});
    shortOpts += "H";
  }
  if (supportedFlags & ARG_INSTRUMENTED) {
    opts.push_back({"instrumented", no_argument, 0, 'I'});
    shortOpts += "I";
  }
  if (supportedFlags & ARG_NUM_FRAMES) {
    opts.push_back({"num-frames", required_argument, 0, 'n'});
    shortOpts += "n:";
  }
  if (supportedFlags & ARG_PLATFORM) {
    opts.push_back({"platform", required_argument, 0, 'p'});
    shortOpts += "p:";
  }
  if (supportedFlags & ARG_SPEED) {
    opts.push_back({"speed", required_argument, 0, 's'});
    shortOpts += "s:";
  }
  if (supportedFlags & ARG_SEG) {
    opts.push_back({"seg", no_argument, 0, 'S'});
    shortOpts += "S";
  }
  if (supportedFlags & ARG_COMPUTE) {
    opts.push_back({"compute", required_argument, 0, 'C'});
    shortOpts += "C:";
  }
  if (supportedFlags & ARG_DETECTIONS) {
    opts.push_back({"detections", no_argument, 0, 'D'});
    shortOpts += "D";
  }
  if (supportedFlags & ARG_COLOR_MODE) {
    opts.push_back({"color-mode", required_argument, 0, 'M'});
    shortOpts += "M:";
  }
  if (supportedFlags & ARG_SAVE_FRAME) {
    opts.push_back({"save-frame", required_argument, 0, 'F'});
    opts.push_back({"save-frame-delay", required_argument, 0, 'f'});
    shortOpts += "F:f:";
  }

  // Sentinel
  opts.push_back({0, 0, 0, 0});

  // Reset getopt for reentrant use
  optind = 1;

  int c;
  while ((c = getopt_long(argc, argv, shortOpts.c_str(), opts.data(), NULL)) != -1) {
    switch (c) {
      case 'h': {
        std::cout << binaryDesc << "\n\n"
                  << "Usage: " << argv[0] << " -m MODEL [options]\n\n"
                  << "Options:\n"
                  << "  -h, --help              Show this help\n";
        if (supportedFlags & ARG_MODEL)
          std::cout << "  -m, --model PATH        Model file [required]\n";
        if (supportedFlags & ARG_PLATFORM)
          std::cout << "  -p, --platform NAME     Platform: imx95 or imx8mp [required]\n";
        if (supportedFlags & ARG_CAMERA)
          std::cout << "  -c, --camera DEVICE     Camera device (default: platform-specific)\n";
        if (supportedFlags & ARG_VIDEO)
          std::cout << "  -v, --video FILE        Video file input (H.264 MP4)\n";
        if (supportedFlags & ARG_IMAGE)
          std::cout << "  -i, --image FILE        Static image input (JPEG)\n";
        if (supportedFlags & ARG_SPEED)
          std::cout << "  -s, --speed RATE        Video playback speed (0.25=quarter, 0.5=half, 1.0=normal)\n";
        if (supportedFlags & ARG_HEADLESS)
          std::cout << "  -H, --headless          No display output\n";
        if (supportedFlags & ARG_INSTRUMENTED)
          std::cout << "  -I, --instrumented      Enable detailed timing breakdown\n";
        if (supportedFlags & ARG_NUM_FRAMES)
          std::cout << "  -n, --num-frames N      Stop after N frames (0=infinite, default=0)\n";
        if (supportedFlags & ARG_SEG)
          std::cout << "  -S, --seg               Instance segmentation mode (YOLOv8-seg model)\n";
        if (supportedFlags & ARG_COMPUTE)
          std::cout << "  -C, --compute BACKEND   edgefirstcameraadaptor backend: auto|opengl|g2d|cpu (default: auto)\n";
        if (supportedFlags & ARG_DETECTIONS)
          std::cout << "  -D, --detections        Print detection details per frame\n";
        if (supportedFlags & ARG_COLOR_MODE)
          std::cout << "  -M, --color-mode MODE   Overlay coloring: class|instance|track (default: class)\n";
        if (supportedFlags & ARG_SAVE_FRAME)
          std::cout << "  -F, --save-frame PATH   Save one frame as PNG (for screenshots/QA)\n"
                    << "  -f, --save-frame-delay N Frame number at which to save (default: 750)\n";
        return 1;
      }
      case 'm': args.model = optarg; break;
      case 'c': args.camera = optarg; break;
      case 'v': args.video = optarg; break;
      case 'i': args.image = optarg; break;
      case 'H': args.headless = true; break;
      case 'I': args.instrumented = true; break;
      case 'n': args.numFrames = atoi(optarg); break;
      case 's': args.speed = atof(optarg); break;
      case 'p': args.platformStr = optarg; break;
      case 'S': args.segmentation = true; break;
      case 'C': args.compute = optarg; break;
      case 'D': args.detections = true; break;
      case 'M': args.colorMode = optarg; break;
      case 'F': args.saveFrame = optarg; break;
      case 'f': args.saveFrameDelay = atoi(optarg); break;
      case '?':
        return -1;
    }
  }
  return 0;
}


/* ─── Bus Callback + SIGINT Handler ───────────────────────────────── */

void commonBusCallback(GstBus *, GstMessage *message, gpointer user_data)
{
  BusCallbackCtx *ctx = (BusCallbackCtx *)user_data;
  switch (GST_MESSAGE_TYPE(message)) {
    case GST_MESSAGE_ERROR: {
      GError *err;
      gchar *debugInfo;
      gst_message_parse_error(message, &err, &debugInfo);
      g_printerr("ERROR from %s: %s\n", GST_OBJECT_NAME(message->src), err->message);
      if (debugInfo)
        g_printerr("Debug: %s\n", debugInfo);
      g_error_free(err);
      g_free(debugInfo);
      g_main_loop_quit(ctx->loop);
      break;
    }
    case GST_MESSAGE_EOS:
      if (ctx->videoLoop) {
        if (ctx->videoRate != 1.0) {
          gst_element_seek(ctx->pipeline, ctx->videoRate, GST_FORMAT_TIME,
                           (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT),
                           GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, 0);
        } else {
          gst_element_seek_simple(ctx->pipeline, GST_FORMAT_TIME,
                                  (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT), 0);
        }
      } else {
        g_print("End-Of-Stream.\n");
        if (ctx->printTiming)
          ctx->printTiming(ctx->appData);
        g_main_loop_quit(ctx->loop);
      }
      break;
    case GST_MESSAGE_STATE_CHANGED: {
      GstState oldState, newState, pendingState;
      gst_message_parse_state_changed(message, &oldState, &newState, &pendingState);
      if (GST_MESSAGE_SRC(message) == GST_OBJECT(ctx->pipeline) &&
          oldState != newState) {
        if (ctx->startedOnce == NULL || !*ctx->startedOnce) {
          g_print("Pipeline: %s -> %s\n",
                  gst_element_state_get_name(oldState),
                  gst_element_state_get_name(newState));
        }
        if (newState == GST_STATE_PLAYING) {
          if (ctx->playing)
            *ctx->playing = TRUE;
          if (ctx->startedOnce && !*ctx->startedOnce && ctx->videoRate != 1.0) {
            gst_element_seek(ctx->pipeline, ctx->videoRate, GST_FORMAT_TIME,
                             (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT),
                             GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, 0);
          }
          if (ctx->startedOnce)
            *ctx->startedOnce = true;
        } else {
          if (ctx->playing)
            *ctx->playing = FALSE;
        }
      }
      break;
    }
    default:
      break;
  }
}

gboolean commonSigintHandler(gpointer user_data)
{
  BusCallbackCtx *ctx = (BusCallbackCtx *)user_data;
  g_print("\nSIGINT — stopping.\n");
  if (ctx->printTiming)
    ctx->printTiming(ctx->appData);
  g_main_loop_quit(ctx->loop);
  return TRUE;
}


/* ─── Throughput Tracker ──────────────────────────────────────────── */

void ThroughputTracker::reset() {
  metric.reset();
  firstFrame = true;
}

void ThroughputTracker::tick(const struct timeval &now) {
  if (!firstFrame)
    metric.record(timeDiffMs(lastFrameTime, now));
  lastFrameTime = now;
  firstFrame = false;
}


/* ─── PTS-Correlated E2E Tracker ─────────────────────────────────── */

void PtsTracker::init() {
  g_mutex_init(&mutex);
}

void PtsTracker::destroy() {
  g_mutex_clear(&mutex);
}

void PtsTracker::recordStart(GstBuffer *buffer, const struct timeval &now) {
  if (!buffer) return;
  GstClockTime pts = GST_BUFFER_PTS(buffer);
  if (GST_CLOCK_TIME_IS_VALID(pts)) {
    g_mutex_lock(&mutex);
    startTimes[pts] = now;
    while (startTimes.size() > 100)
      startTimes.erase(startTimes.begin());
    g_mutex_unlock(&mutex);
  }
}

bool PtsTracker::consumeStart(GstBuffer *buffer, struct timeval &startTime) {
  if (!buffer) return false;
  GstClockTime pts = GST_BUFFER_PTS(buffer);
  if (!GST_CLOCK_TIME_IS_VALID(pts)) return false;

  g_mutex_lock(&mutex);
  auto it = startTimes.find(pts);
  if (it != startTimes.end()) {
    startTime = it->second;
    startTimes.erase(it);
    g_mutex_unlock(&mutex);
    return true;
  }
  g_mutex_unlock(&mutex);
  return false;
}


/* ─── Inference Latency Query ─────────────────────────────────────── */

void queryInferenceLatency(GstElement *tensorFilter, TimingMetric &metric) {
  if (!tensorFilter) return;
  gint64 latencyUs = 0;
  g_object_get(tensorFilter, "latency", &latencyUs, NULL);
  if (latencyUs > 0)
    metric.record(latencyUs / 1000.0);
}


/* ─── PipelineProbes: per-element + full pipeline latency ─────────── */

static GstPadProbeReturn
pp_queueSinkProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  PipelineProbes *p = (PipelineProbes *)user_data;
  gettimeofday(&p->queueSinkStart, NULL);
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
pp_queueSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  PipelineProbes *p = (PipelineProbes *)user_data;
  gettimeofday(&p->queueSrcEnd, NULL);
  p->queueDwell.record(timeDiffMs(p->queueSinkStart, p->queueSrcEnd));
  // Race-safe snapshot for fullLatency: we're now on the downstream
  // thread, and the buffer is committed to this processing cycle.
  // queueSinkStart can be safely overwritten by upstream after this
  // assignment without affecting the in-flight buffer's measurement.
  p->currentFrameQueueStart = p->queueSinkStart;
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
pp_preprocSinkProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  PipelineProbes *p = (PipelineProbes *)user_data;
  gettimeofday(&p->preprocStart, NULL);
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
pp_preprocSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  PipelineProbes *p = (PipelineProbes *)user_data;
  gettimeofday(&p->preprocEnd, NULL);
  p->preproc.record(timeDiffMs(p->preprocStart, p->preprocEnd));
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
pp_filterSinkProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  PipelineProbes *p = (PipelineProbes *)user_data;
  gettimeofday(&p->filterSinkStart, NULL);
  return GST_PAD_PROBE_OK;
}

static GstPadProbeReturn
pp_filterSrcProbe(GstPad *, GstPadProbeInfo *, gpointer user_data)
{
  PipelineProbes *p = (PipelineProbes *)user_data;
  gettimeofday(&p->filterSrcEnd, NULL);
  p->filterElement.record(timeDiffMs(p->filterSinkStart, p->filterSrcEnd));
  return GST_PAD_PROBE_OK;
}

void PipelineProbes::reset()
{
  queueDwell.reset();
  preproc.reset();
  filterElement.reset();
  inference.reset();
  postproc.reset();
  fullLatency.reset();
  queueSinkStart = {};
  queueSrcEnd = {};
  preprocStart = {};
  preprocEnd = {};
  filterSinkStart = {};
  filterSrcEnd = {};
  currentFrameQueueStart = {};
  tensorFilter = nullptr;
}

/** Helper: install a probe pair (sink+src) on a named element. */
static void install_pad_probe(GstElement *pipeline, const char *elementName,
                              const char *padName, GstPadProbeCallback cb,
                              gpointer user_data)
{
  if (!elementName) return;
  GstElement *elem = gst_bin_get_by_name(GST_BIN(pipeline), elementName);
  if (!elem) {
    g_printerr("PipelineProbes: element '%s' not found\n", elementName);
    return;
  }
  GstPad *pad = gst_element_get_static_pad(elem, padName);
  if (pad) {
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, cb, user_data, NULL);
    gst_object_unref(pad);
  }
  gst_object_unref(elem);
}

bool PipelineProbes::install(GstElement *pipeline,
                             const char *queueName,
                             const char *preprocName,
                             const char *filterName)
{
  if (!pipeline) return false;

  install_pad_probe(pipeline, queueName,   "sink", pp_queueSinkProbe,   this);
  install_pad_probe(pipeline, queueName,   "src",  pp_queueSrcProbe,    this);
  install_pad_probe(pipeline, preprocName, "sink", pp_preprocSinkProbe, this);
  install_pad_probe(pipeline, preprocName, "src",  pp_preprocSrcProbe,  this);
  install_pad_probe(pipeline, filterName,  "sink", pp_filterSinkProbe,  this);
  install_pad_probe(pipeline, filterName,  "src",  pp_filterSrcProbe,   this);

  // Cache tensor_filter ref for inference latency queries.
  if (filterName) {
    tensorFilter = gst_bin_get_by_name(GST_BIN(pipeline), filterName);
  }
  return true;
}

void PipelineProbes::teardown()
{
  if (tensorFilter) {
    gst_object_unref(tensorFilter);
    tensorFilter = nullptr;
  }
}

void PipelineProbes::recordPost(const struct timeval &postStart,
                                const struct timeval &postEnd)
{
  postproc.record(timeDiffMs(postStart, postEnd));

  // Per-buffer full pipeline latency, race-safe via the snapshot taken
  // in the queue src probe (downstream thread). currentFrameQueueStart
  // is stable for the lifetime of this buffer's downstream processing.
  if (currentFrameQueueStart.tv_sec != 0)
    fullLatency.record(timeDiffMs(currentFrameQueueStart, postEnd));
}

void PipelineProbes::recordInference()
{
  queryInferenceLatency(tensorFilter, inference);
}

static void pp_print_metric(const char *label, const TimingMetric &m)
{
  if (m.count > 0) {
    printf("     %-52s  avg %7.3f ms  min %7.3f ms  max %7.3f ms  [%d]\n",
           label, m.avg(), m.minMs, m.maxMs, m.count);
  } else {
    printf("     %-52s  (no samples)\n", label);
  }
}

void PipelineProbes::printReport(const char *title) const
{
  printf("\n");
  printf("==============================================================================\n");
  printf("  %s\n", title ? title : "Pipeline timing report");
  printf("==============================================================================\n");

  printf("\n  PER-ELEMENT PIPELINE LATENCY (pad-probe measured)\n");
  printf("  --------------------------------------------------------------------------\n");
  pp_print_metric("queue (thread-nn dwell)",            queueDwell);
  pp_print_metric("edgefirstcameraadaptor (preproc)",   preproc);
  pp_print_metric("tensor_filter element (GST total)",  filterElement);
  pp_print_metric("  └─ NPU Invoke (internal latency)", inference);
  pp_print_metric("tensor_sink callback (post-process)",postproc);

  printf("\n  FULL PIPELINE LATENCY (queue sink → end of post-processing)\n");
  printf("  --------------------------------------------------------------------------\n");
  pp_print_metric("full per-buffer latency",            fullLatency);

  printf("\n==============================================================================\n");
}


/* ─── Source Element Construction ─────────────────────────────────── */

InputSource determineInputSource(const ParsedArgs &args, bool usesLibcamerasrc) {
  if (!args.image.empty()) return INPUT_IMAGE;
  if (!args.video.empty()) return INPUT_VIDEO;
  if (usesLibcamerasrc) return INPUT_CAMERA_LIBCAMERA;
  return INPUT_CAMERA_V4L2;
}

char *buildSourceElement(InputSource source, const ParsedArgs &args,
                         int srcWidth, int srcHeight, int numBuffers)
{
  switch (source) {
    case INPUT_IMAGE:
      // Use videoscale+videoconvert (software) before imagefreeze so the
      // repeated frames are in system memory.  imxvideoconvert_g2d (HW)
      // before imagefreeze produces DMA-BUF backed buffers that imagefreeze
      // cannot re-negotiate with downstream HW elements on i.MX 8M Plus.
      return g_strdup_printf(
          "filesrc location=%s ! jpegdec ! videoscale ! videoconvert ! "
          "video/x-raw,format=NV12,width=%d,height=%d ! imagefreeze",
          args.image.c_str(), srcWidth, srcHeight);

    case INPUT_VIDEO:
      return g_strdup_printf(
          "filesrc location=%s ! qtdemux ! h264parse ! v4l2h264dec",
          args.video.c_str());

    case INPUT_CAMERA_LIBCAMERA:
      return g_strdup_printf(
          "libcamerasrc ! video/x-raw,format=YUY2,width=%d,height=%d",
          srcWidth, srcHeight);

    case INPUT_CAMERA_V4L2:
      if (numBuffers > 0) {
        return g_strdup_printf(
            "v4l2src device=%s num-buffers=%d ! "
            "video/x-raw,format=YUY2,width=%d,height=%d,framerate=30/1",
            args.camera.c_str(), numBuffers, srcWidth, srcHeight);
      }
      return g_strdup_printf(
          "v4l2src device=%s ! "
          "video/x-raw,format=YUY2,width=%d,height=%d,framerate=30/1",
          args.camera.c_str(), srcWidth, srcHeight);
  }
  return g_strdup("fakesrc");  // Unreachable
}


/* ─── i.MX 95 Environment Setup ──────────────────────────────────── */

void setupImx95Environment(bool neutron_zero_copy) {
  const char *pm = getenv("LIBCAMERA_PIPELINES_MATCH_LIST");
  if (!pm || strlen(pm) == 0) {
    g_print("Setting LIBCAMERA_PIPELINES_MATCH_LIST='nxp/neo,imx8-isi'\n");
    setenv("LIBCAMERA_PIPELINES_MATCH_LIST", "nxp/neo,imx8-isi", 1);
  }
  /* Each pipeline hard-codes its zero-copy requirement: yolov8n_imx95 uses
   * NnsDmaBufInputPool (edgefirstcameraadaptor writes to the Neutron DMA-BUF
   * directly, so zero-copy must be enabled).  All other pipelines copy the
   * tensor into Neutron's DMA-BUF internally. */
  const char *val = neutron_zero_copy ? "1" : "0";
  g_print("Setting NEUTRON_ENABLE_ZERO_COPY=%s\n", val);
  setenv("NEUTRON_ENABLE_ZERO_COPY", val, 1);
}
