/**
 * Copyright 2026 EdgeFirst AI (Au-Zone Technologies)
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * YOLOv8-seg Instance Segmentation — Ara-2 NPU + EdgeFirst HAL
 *
 * Demonstrates the appsink/appsrc bridge pattern for GPU-accelerated
 * segmentation mask overlay rendering in a GStreamer pipeline.
 *
 * Architecture:
 *   source → tee
 *     tee → queue → appsink (capture camera DMA-BUF frames)
 *     tee → queue → edgefirstcameraadaptor → tensor_filter(ara2) → tensor_sink
 *   appsrc → waylandsink (HAL-rendered RGBA canvas)
 *
 * In the tensor_sink callback:
 *   1. Pull latest camera frame from appsink (inode-cached hal_import_image)
 *   2. Fused hal_image_processor_draw_masks: decode + render in one call
 *      with camera frame as background → colored mask overlay on live video
 *   3. Push canvas DMA-BUF to appsrc → waylandsink
 *
 * This mirrors the architecture of ara2-rs/examples/yolov8_live.rs but
 * wired into GStreamer for integration with NNStreamer pipelines.
 */

#include <gst/gst.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <glib.h>
#include <glib-unix.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

#include "logging.hpp"
#include "common/yolov8_common.hpp"

#include <edgefirst/hal.h>


#define MAX_TENSORS 16


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


/* ─── Application data ───────────────────────────────────────────── */

struct AppData {
  GstElement *pipeline;
  GMainLoop *loop;
  GstBus *bus;
  gboolean playing;

  BusCallbackCtx busCtx;
  ThroughputTracker throughput;

  TimingMetric preproc;
  TimingMetric inference;
  TimingMetric postproc;
  struct timeval preprocStart;

  int totalDetections;
  int framesWithDetections;
  int totalFrames;
  int maxFrames;

  GstElement *tensorFilter;
  GstElement *tensorSink;
  GstElement *cameraadaptor;
  LetterboxParams letterbox;

  bool headless;

  /* HAL decoder + image processor */
  hal_decoder *decoder;
  hal_image_processor *processor;


  /* Tensor metadata from caps */
  int tensor_count;
  hal_dtype tensor_dtypes[MAX_TENSORS];
  size_t tensor_shapes[MAX_TENSORS][8];   /* NNStreamer squeezed shapes */
  size_t tensor_ndims[MAX_TENSORS];
  size_t hal_shapes[MAX_TENSORS][8];      /* HAL-convention shapes */
  size_t hal_ndims[MAX_TENSORS];
  bool decoder_configured;

  /* Display: appsink captures camera frames, appsrc pushes rendered canvas */
  hal_pixel_format srcPixelFormat;  /* detected from appsink caps */
  GstElement *appsink;     /* captures camera DMA-BUF frames */
  GstElement *appsrc;      /* pushes rendered RGBA canvas */
  GstAllocator *dmabufAlloc;

  /* Double-buffered canvas: render into one while waylandsink displays the other.
   * Each push does dup(canvasFd) — GStreamer takes ownership of the dup'd fd
   * and closes it on buffer unref. Our canvasFd stays alive for reuse. */
  static const int NUM_CANVAS = 2;
  hal_tensor *canvas[2];
  int canvasFd[2];
  size_t canvasSize;
  int canvasIdx;

  /* Camera frame import cache — keyed by inode, reuse across frames */
  std::map<ino_t, hal_tensor *> frameTensorCache;

  int srcWidth, srcHeight;
};


/* ─── NNStreamer type/dim parsing ─────────────────────────────────── */

static hal_dtype nnstreamer_type_to_hal(const char *type_str)
{
  if (!type_str) return HAL_DTYPE_F32;
  if (g_strcmp0(type_str, "float32") == 0) return HAL_DTYPE_F32;
  if (g_strcmp0(type_str, "uint8")   == 0) return HAL_DTYPE_U8;
  if (g_strcmp0(type_str, "int8")    == 0) return HAL_DTYPE_I8;
  if (g_strcmp0(type_str, "int16")   == 0) return HAL_DTYPE_I16;
  if (g_strcmp0(type_str, "int32")   == 0) return HAL_DTYPE_I32;
  return HAL_DTYPE_F32;
}

static size_t parse_nnstreamer_dims(const char *dim_str, size_t *shape, size_t max_ndim)
{
  gchar **parts = g_strsplit(dim_str, ":", (gint)max_ndim + 1);
  size_t n = 0;
  while (parts[n] && n < max_ndim) n++;
  while (n > 1 && g_strcmp0(parts[n - 1], "1") == 0) n--;
  size_t raw[8];
  for (size_t i = 0; i < n && i < 8; i++)
    raw[i] = (size_t)g_ascii_strtoull(parts[n - 1 - i], NULL, 10);
  g_strfreev(parts);
  size_t out = 0;
  for (size_t i = 0; i < n; i++) {
    if (raw[i] != 1 || out == 0)
      shape[out++] = raw[i];
  }
  if (out < 2 && n >= 2)
    shape[out++] = 1;
  return out;
}


/* ─── Auto-configure HAL decoder from NNStreamer caps + quant meta ── */

static bool auto_config_decoder(AppData *app, GstCaps *caps, GstBuffer *buffer)
{
  GstStructure *s = gst_caps_get_structure(caps, 0);
  gint num_tensors = 0;
  const gchar *dims_str = gst_structure_get_string(s, "dimensions");
  const gchar *types_str = gst_structure_get_string(s, "types");

  if (!gst_structure_get_int(s, "num_tensors", &num_tensors) ||
      num_tensors <= 0 || !dims_str || !types_str)
    return false;

  gchar **dim_parts  = g_strsplit(dims_str,  ",", num_tensors + 1);
  gchar **type_parts = g_strsplit(types_str, ",", num_tensors + 1);

  app->tensor_count = (num_tensors < MAX_TENSORS) ? num_tensors : MAX_TENSORS;

  bool has_protos = false;
  size_t proto_channels = 0;

  for (int i = 0; i < app->tensor_count; i++) {
    app->tensor_ndims[i] = parse_nnstreamer_dims(dim_parts[i], app->tensor_shapes[i], 8);
    app->tensor_dtypes[i] = nnstreamer_type_to_hal(type_parts[i]);
    if (app->tensor_ndims[i] == 3) {
      has_protos = true;
      proto_channels = app->tensor_shapes[i][2];
    }
  }

  /* Identify tensor roles */
  int protos_idx = -1, scores_idx = -1, boxes_idx = -1, coeffs_idx = -1;
  for (int i = 0; i < app->tensor_count; i++) {
    size_t ndim = app->tensor_ndims[i];
    if (ndim == 3) protos_idx = i;
    else if (ndim == 2 && app->tensor_shapes[i][1] == 4) boxes_idx = i;
    else if (ndim == 2 && has_protos && app->tensor_shapes[i][1] == proto_channels) coeffs_idx = i;
    else if (scores_idx < 0) scores_idx = i;
  }
  if (coeffs_idx < 0 && protos_idx >= 0) {
    for (int i = 0; i < app->tensor_count; i++) {
      if (i == protos_idx || i == boxes_idx || i == scores_idx) continue;
      if (app->tensor_ndims[i] == 2 && app->tensor_shapes[i][1] == proto_channels)
        { coeffs_idx = i; break; }
    }
  }

  log_info("Auto-config: %d tensors, proto_channels=%zu\n", app->tensor_count, proto_channels);
  for (int i = 0; i < app->tensor_count; i++) {
    const char *role = (i == protos_idx) ? "PROTOS" : (i == boxes_idx) ? "BOXES" :
                       (i == scores_idx) ? "SCORES" : (i == coeffs_idx) ? "MASK_COEFF" : "?";
    log_info("  [%d] %-11s ndim=%zu dtype=%d\n", i, role, app->tensor_ndims[i], app->tensor_dtypes[i]);
  }

  g_strfreev(dim_parts);
  g_strfreev(type_parts);

  if (boxes_idx < 0 || scores_idx < 0) { log_error("Cannot identify boxes/scores\n"); return false; }

  /* Compute HAL-convention shapes */
  for (int i = 0; i < app->tensor_count; i++) {
    size_t *ns = app->tensor_shapes[i];
    if (i == protos_idx) {
      app->hal_shapes[i][0] = 1; app->hal_shapes[i][1] = ns[2];
      app->hal_shapes[i][2] = ns[0]; app->hal_shapes[i][3] = ns[1];
      app->hal_ndims[i] = 4;
    } else {
      app->hal_shapes[i][0] = 1; app->hal_shapes[i][1] = ns[1]; app->hal_shapes[i][2] = ns[0];
      app->hal_ndims[i] = 3;
    }
  }

  /* Extract quant params */
  QuantParams qp[MAX_TENSORS] = {};
  for (int i = 0; i < app->tensor_count; i++) {
    if (!extractQuantParams(buffer, i, qp[i])) { qp[i].scale = 1.0; qp[i].zeroPoint = 0; }
  }
  if (boxes_idx >= 0) qp[boxes_idx].scale /= (double)MODEL_INPUT_SIZE;

  /* Build decoder */
  hal_decoder_params *params = hal_decoder_params_new();
  hal_decoder_params_set_score_threshold(params, CONF_THRESHOLD);
  hal_decoder_params_set_iou_threshold(params, NMS_IOU_THRESHOLD);
  hal_decoder_params_set_nms(params, HAL_NMS_CLASS_AGNOSTIC);

  auto add_split = [&](int idx, HalOutputType type, HalDimName d1) -> int {
    size_t shape[3] = {1, app->tensor_shapes[idx][1], app->tensor_shapes[idx][0]};
    HalDimName dims[3] = {HAL_DIM_NAME_BATCH, d1, HAL_DIM_NAME_NUM_BOXES};
    int r = hal_decoder_params_add_output(params, type, HAL_DECODER_TYPE_ULTRALYTICS, shape, dims, 3);
    if (r >= 0) hal_decoder_params_output_set_quantization(params, r, (float)qp[idx].scale, (int)qp[idx].zeroPoint);
    if (type == HAL_OUTPUT_TYPE_BOXES && r >= 0) hal_decoder_params_output_set_normalized(params, r, 1);
    return r;
  };

  add_split(scores_idx, HAL_OUTPUT_TYPE_SCORES, HAL_DIM_NAME_NUM_CLASSES);
  add_split(boxes_idx, HAL_OUTPUT_TYPE_BOXES, HAL_DIM_NAME_BOX_COORDS);
  if (coeffs_idx >= 0)
    add_split(coeffs_idx, HAL_OUTPUT_TYPE_MASK_COEFFICIENTS, HAL_DIM_NAME_NUM_PROTOS);

  if (protos_idx >= 0) {
    size_t *ns = app->tensor_shapes[protos_idx];
    size_t shape[4] = {1, ns[2], ns[0], ns[1]};
    HalDimName dims[4] = {HAL_DIM_NAME_BATCH, HAL_DIM_NAME_NUM_PROTOS, HAL_DIM_NAME_HEIGHT, HAL_DIM_NAME_WIDTH};
    int pi = hal_decoder_params_add_output(params, HAL_OUTPUT_TYPE_PROTOS, HAL_DECODER_TYPE_ULTRALYTICS, shape, dims, 4);
    if (pi >= 0) hal_decoder_params_output_set_quantization(params, pi, (float)qp[protos_idx].scale, (int)qp[protos_idx].zeroPoint);
  }

  app->decoder = hal_decoder_new(params);
  hal_decoder_params_free(params);
  if (!app->decoder) { log_error("HAL decoder creation failed (errno=%d)\n", errno); return false; }

  char *mt = hal_decoder_model_type(app->decoder);
  log_info("HAL decoder: type=%s, normalized=%d\n", mt ? mt : "?", hal_decoder_normalized_boxes(app->decoder));
  free(mt);

  app->decoder_configured = true;
  return true;
}


/* ─── Import camera frame as HAL tensor (inode-cached) ───────────── */

static hal_tensor *import_camera_frame(AppData *app, GstBuffer *buf)
{
  GstMemory *mem = gst_buffer_peek_memory(buf, 0);
  if (!gst_is_dmabuf_memory(mem)) return NULL;

  int fd = gst_dmabuf_memory_get_fd(mem);
  struct stat st;
  if (fstat(fd, &st) != 0) return NULL;

  /* Cache lookup — same inode = same physical DMA-BUF buffer.
   * Content changes between frames (decoder writes new data) but the
   * EGLImage handle remains valid — it's a pointer to live memory. */
  auto it = app->frameTensorCache.find(st.st_ino);
  if (it != app->frameTensorCache.end()) return it->second;

  /* Cache miss — import via PlaneDescriptor */
  GstVideoMeta *vmeta = gst_buffer_get_video_meta(buf);

  hal_plane_descriptor *pd = hal_plane_descriptor_new(fd);
  if (!pd) return NULL;

  if (vmeta && vmeta->n_planes >= 1 && vmeta->stride[0] > 0)
    hal_plane_descriptor_set_stride(pd, vmeta->stride[0]);

  hal_plane_descriptor *chroma = NULL;
  if (app->srcPixelFormat == HAL_PIXEL_FORMAT_NV12) {
    guint n_mem = gst_buffer_n_memory(buf);
    if (n_mem >= 2) {
      GstMemory *mem1 = gst_buffer_peek_memory(buf, 1);
      if (gst_is_dmabuf_memory(mem1)) {
        int uv_fd = gst_dmabuf_memory_get_fd(mem1);
        chroma = hal_plane_descriptor_new(uv_fd);
        if (chroma) {
          if (vmeta && vmeta->n_planes >= 2) {
            if (vmeta->offset[1] > 0)
              hal_plane_descriptor_set_offset(chroma, vmeta->offset[1]);
            if (vmeta->stride[1] > 0)
              hal_plane_descriptor_set_stride(chroma, vmeta->stride[1]);
          }
        }
      }
    }
  }

  hal_tensor *t = hal_import_image(app->processor, pd, chroma,
      app->srcWidth, app->srcHeight, app->srcPixelFormat, HAL_DTYPE_U8);
  if (t) app->frameTensorCache[st.st_ino] = t;
  return t;
}


/* ─── Preprocessing timing probes ────────────────────────────────── */

static GstPadProbeReturn preprocStartProbe(GstPad *, GstPadProbeInfo *, gpointer ud)
{ ((AppData *)ud)->preprocStart = {}; gettimeofday(&((AppData *)ud)->preprocStart, NULL); return GST_PAD_PROBE_OK; }

static GstPadProbeReturn preprocEndProbe(GstPad *, GstPadProbeInfo *, gpointer ud)
{ struct timeval e; gettimeofday(&e, NULL); ((AppData *)ud)->preproc.record(timeDiffMs(((AppData *)ud)->preprocStart, e)); return GST_PAD_PROBE_OK; }


/* ─── Timing report ──────────────────────────────────────────────── */

static void printTiming(void *userData)
{
  AppData *app = (AppData *)userData;
  printf("\n==============================================================================\n");
  printf("  YOLOV8-SEG — ARA-2 NPU + EDGEFIRST HAL (fused draw_masks)\n");
  printf("==============================================================================\n");

  if (app->preproc.count > 0)
    printf("\n  Preprocess:  %7.3f ms avg  [%d frames]\n", app->preproc.avg(), app->preproc.count);
  if (app->inference.count > 0)
    printf("  Inference:   %7.3f ms avg  [%d samples]\n", app->inference.avg(), app->inference.count);
  if (app->postproc.count > 0)
    printf("  Draw masks:  %7.3f ms avg  [%d frames]\n", app->postproc.avg(), app->postproc.count);
  if (app->throughput.metric.count > 0) {
    double avg = app->throughput.metric.avg();
    printf("  End-to-end:  %7.3f ms (%5.1f FPS)  [%d frames]\n", avg, 1000.0/avg, app->throughput.metric.count);
  }
  printf("\n  Detections: %d total, %d/%d frames with detections\n",
         app->totalDetections, app->framesWithDetections, app->totalFrames);
  printf("==============================================================================\n");
}


/* ─── tensor_sink new-data callback ──────────────────────────────── */

static void newDataCallback(GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  struct timeval startTime, endTime;
  gettimeofday(&startTime, NULL);
  AppData *app = (AppData *)user_data;

  /* Query letterbox on first frame */
  if (app->throughput.firstFrame && app->cameraadaptor) {
    gfloat scale = 0; gint top = 0, left = 0;
    g_object_get(app->cameraadaptor, "letterbox-scale", &scale,
                 "letterbox-top", &top, "letterbox-left", &left, NULL);
    app->letterbox.scale = scale; app->letterbox.padX = left; app->letterbox.padY = top;
    log_info("Letterbox: scale=%.4f top=%d left=%d\n", scale, top, left);
  }

  app->throughput.tick(startTime);
  queryInferenceLatency(app->tensorFilter, app->inference);

  /* Auto-configure decoder on first frame */
  if (!app->decoder_configured) {
    GstPad *pad = gst_element_get_static_pad(element, "sink");
    GstCaps *caps = pad ? gst_pad_get_current_caps(pad) : NULL;
    if (caps) {
      if (!auto_config_decoder(app, caps, buffer)) {
        log_error("Decoder auto-config failed\n");
        gst_caps_unref(caps); gst_object_unref(pad);
        g_main_loop_quit(app->loop); return;
      }
      gst_caps_unref(caps);
    }
    if (pad) gst_object_unref(pad);
  }
  if (!app->decoder) return;

  /* Validate buffer */
  guint n_mem = gst_buffer_n_memory(buffer);
  if ((int)n_mem != app->tensor_count) return;

  /* Wrap NN output tensors — fresh each frame, freed after draw_masks.
   * HAL dups fds internally. No cache needed. */
  hal_tensor *hal_outputs[MAX_TENSORS] = {};
  bool tensor_wrap_ok = true;
  for (int j = 0; j < app->tensor_count; j++) {
    GstMemory *mem = gst_buffer_peek_memory(buffer, j);
    if (!gst_is_dmabuf_memory(mem)) {
      log_error("Tensor %d not DMA-BUF\n", j);
      tensor_wrap_ok = false;
      break;
    }
    int fd = gst_dmabuf_memory_get_fd(mem);
    hal_outputs[j] = hal_tensor_from_fd(app->tensor_dtypes[j], fd,
        app->hal_shapes[j], app->hal_ndims[j], NULL);
    if (!hal_outputs[j]) {
      log_error("tensor_from_fd failed tensor %d\n", j);
      tensor_wrap_ok = false;
      break;
    }
  }
  if (!tensor_wrap_ok) {
    for (int j = 0; j < app->tensor_count; j++)
      if (hal_outputs[j]) hal_tensor_free(hal_outputs[j]);
    return;
  }

  /* Pull latest camera frame from appsink for background */
  hal_tensor *background = NULL;
  if (app->appsink) {
    GstSample *sample = gst_app_sink_try_pull_sample(GST_APP_SINK(app->appsink), 0);
    if (sample) {
      GstBuffer *frameBuf = gst_sample_get_buffer(sample);
      if (frameBuf)
        background = import_camera_frame(app, frameBuf);
      gst_sample_unref(sample);
    }
  }

  /* Letterbox in normalized coordinates */
  float lb[4] = {0};
  if (app->letterbox.scale > 0) {
    lb[0] = (float)app->letterbox.padX / MODEL_INPUT_SIZE;
    lb[1] = (float)app->letterbox.padY / MODEL_INPUT_SIZE;
    lb[2] = 1.0f - lb[0];
    lb[3] = 1.0f - lb[1];
  }

  /* Double-buffer: render into current canvas, push to appsrc, swap */
  int ci = app->canvasIdx;
  hal_detect_box_list *boxes = NULL;
  if (app->processor && app->canvas[ci]) {
    int ret = hal_image_processor_draw_masks(
        app->processor, app->decoder,
        (const hal_tensor *const *)hal_outputs, app->tensor_count,
        app->canvas[ci], background,
        0.5f, /* opacity */
        app->letterbox.scale > 0 ? lb : NULL,
        HAL_COLOR_MODE_INSTANCE,
        &boxes);

    if (ret != 0) {
      static bool first = true;
      if (first) { log_error("draw_masks failed (ret=%d errno=%d)\n", ret, errno); first = false; }
    }
  }

  /* Stats */
  app->totalFrames++;
  if (boxes) {
    size_t n = hal_detect_box_list_len(boxes);
    if (n > 0) { app->framesWithDetections++; app->totalDetections += n; }

    static bool first_print = true;
    if (first_print && n > 0) {
      log_info("First frame: %zu detections\n", n);
      for (size_t d = 0; d < n && d < 5; d++) {
        hal_detect_box box;
        if (hal_detect_box_list_get(boxes, d, &box) == 0) {
          const char *name = (box.label >= 0 && box.label < NUM_CLASSES) ? cocoClassNames[box.label] : "?";
          log_info("  %s %.0f%%\n", name, box.score * 100.0f);
        }
      }
      first_print = false;
    }
    hal_detect_box_list_free(boxes);
  }

  /* Push rendered canvas to appsrc. dup() the fd so GStreamer owns its copy —
   * gst_dmabuf_allocator_alloc takes ownership and closes it on buffer unref.
   * Double-buffer: swap canvas so waylandsink holds one while we render the other. */
  if (app->appsrc && app->canvasFd[ci] >= 0) {
    int pushFd = dup(app->canvasFd[ci]);
    GstMemory *mem = gst_dmabuf_allocator_alloc(app->dmabufAlloc, pushFd, app->canvasSize);
    GstBuffer *outBuf = gst_buffer_new();
    gst_buffer_append_memory(outBuf, mem);
    gst_app_src_push_buffer(GST_APP_SRC(app->appsrc), outBuf);
  }
  app->canvasIdx = 1 - ci;

  /* Free per-frame NN output tensors (NOT cached).
   * Background is cached by inode — do NOT free it here. */
  for (int j = 0; j < app->tensor_count; j++) {
    if (hal_outputs[j]) hal_tensor_free(hal_outputs[j]);
  }

  /* Frame limit */
  if (app->maxFrames > 0 && app->totalFrames >= app->maxFrames)
    g_main_loop_quit(app->loop);

  gettimeofday(&endTime, NULL);
  app->postproc.record(timeDiffMs(startTime, endTime));
}


/* ─── Main ───────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
  ParsedArgs pargs;
  pargs.camera = "";

  uint32_t flags = ARG_MODEL | ARG_CAMERA | ARG_VIDEO | ARG_IMAGE |
                   ARG_HEADLESS | ARG_NUM_FRAMES | ARG_PLATFORM;

  int ret = parseArgs(argc, argv, flags,
      "YOLOv8-seg — Ara-2 NPU + EdgeFirst HAL (appsink/appsrc bridge)", pargs);
  if (ret != 0) return ret > 0 ? 0 : 1;

  if (pargs.model.empty()) { log_error("Provide model path with -m\n"); return 1; }

  Platform platform = PLATFORM_IMX8MP;
  if (!pargs.platformStr.empty()) {
    if (pargs.platformStr == "imx95") platform = PLATFORM_IMX95;
    else if (pargs.platformStr == "imx8mp") platform = PLATFORM_IMX8MP;
    else { log_error("Unknown platform: %s\n", pargs.platformStr.c_str()); return 1; }
  } else { log_error("Provide --platform imx95 or imx8mp\n"); return 1; }

  const PlatformConfig &plat = platformConfigs[platform];
  if (pargs.camera.empty() && plat.defaultCamera) pargs.camera = plat.defaultCamera;
  if (platform == PLATFORM_IMX95) setupImx95Environment(false);

  gst_init(&argc, &argv);

  log_info("YOLOv8-seg for Ara-2 NPU — %s\n", plat.name);
  log_info("Model: %s\n", pargs.model.c_str());
  log_info("Mode: %s\n", pargs.headless ? "headless" : "display");

  InputSource srcType = determineInputSource(pargs, plat.usesLibcamerasrc);
  char *srcStr = buildSourceElement(srcType, pargs);

  /* NN branch */
  char *nnBranch = g_strdup_printf(
      "queue name=thread-nn leaky=2 max-size-buffers=2 ! "
      "edgefirstcameraadaptor name=preproc model-width=%d model-height=%d "
      "model-dtype=int8 model-layout=chw letterbox=true ! "
      "tensor_filter name=tfilter framework=ara2 model=%s "
      "custom=EnableStats:true latency=1 ! "
      "tensor_sink name=tsink",
      MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, pargs.model.c_str());

  /* Build pipeline */
  char *pipelineStr;
  if (pargs.headless) {
    pipelineStr = g_strdup_printf("%s ! %s", srcStr, nnBranch);
  } else {
    /* tee → appsink (camera frames) + NN branch; appsrc → waylandsink */
    pipelineStr = g_strdup_printf(
        "%s ! tee name=t "
        "t. ! queue leaky=2 max-size-buffers=1 ! "
        "appsink name=videosink emit-signals=false drop=true max-buffers=1 sync=true "
        "t. ! %s "
        "appsrc name=display stream-type=0 format=3 is-live=true "
        "do-timestamp=true max-buffers=2 block=false ! "
        "waylandsink async=false",
        srcStr, nnBranch);
  }
  g_free(srcStr);
  g_free(nnBranch);

  log_info("Pipeline: %s\n\n", pipelineStr);

  AppData app = {};
  app.canvasFd[0] = app.canvasFd[1] = -1;
  app.canvasIdx = 0;
  app.maxFrames = pargs.numFrames;
  app.headless = pargs.headless;
  app.preproc.reset(); app.inference.reset(); app.postproc.reset(); app.throughput.reset();

  /* Always create image processor + canvas for draw_masks (decode + render).
   * Headless mode still decodes and renders to the canvas but doesn't push
   * to appsrc — the canvas DMA-BUF is reused frame-to-frame. */
  log_info("Creating HAL image processor (OpenGL)...\n");
  fflush(stdout);
  app.processor = hal_image_processor_new_with_backend(HAL_COMPUTE_BACKEND_OPENGL);
  if (app.processor) {
    log_info("HAL image processor ready\n");
    app.srcWidth = SOURCE_WIDTH;
    app.srcHeight = SOURCE_HEIGHT;
    /* Video files → v4l2h264dec → NV12; camera → YUYV (both platforms) */
    app.srcPixelFormat = (srcType == INPUT_VIDEO)
        ? HAL_PIXEL_FORMAT_NV12 : HAL_PIXEL_FORMAT_YUYV;
    for (int i = 0; i < AppData::NUM_CANVAS; i++) {
      app.canvas[i] = hal_image_processor_create_image(
          app.processor, app.srcWidth, app.srcHeight, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
      if (app.canvas[i]) {
        app.canvasFd[i] = hal_tensor_clone_fd(app.canvas[i]);
        app.canvasSize = hal_tensor_size(app.canvas[i]);
      }
    }
    if (app.canvas[0] && app.canvas[1]) {
      log_info("Canvas: %dx%d RGBA double-buffered, fds=[%d,%d], %zu bytes\n",
               app.srcWidth, app.srcHeight, app.canvasFd[0], app.canvasFd[1], app.canvasSize);
    }
  } else {
    log_error("HAL image processor failed\n");
  }

  bool startedOnce = false;
  app.busCtx.playing = &app.playing;
  app.busCtx.startedOnce = &startedOnce;
  app.busCtx.videoLoop = !pargs.video.empty() && pargs.image.empty();
  app.busCtx.videoRate = 1.0;
  app.busCtx.printTiming = NULL;
  app.busCtx.appData = &app;

  app.loop = g_main_loop_new(NULL, FALSE);
  app.pipeline = gst_parse_launch(pipelineStr, NULL);
  g_free(pipelineStr);
  if (!app.pipeline) { log_error("Pipeline creation failed\n"); return 1; }

  app.busCtx.pipeline = app.pipeline;
  app.busCtx.loop = app.loop;

  app.bus = gst_element_get_bus(app.pipeline);
  gst_bus_add_signal_watch(app.bus);
  g_signal_connect(app.bus, "message", G_CALLBACK(commonBusCallback), &app.busCtx);

  /* tensor_sink */
  app.tensorSink = gst_bin_get_by_name(GST_BIN(app.pipeline), "tsink");
  g_signal_connect(app.tensorSink, "new-data", G_CALLBACK(newDataCallback), &app);


  /* tensor_filter */
  app.tensorFilter = gst_bin_get_by_name(GST_BIN(app.pipeline), "tfilter");

  /* appsink for camera frames + appsrc for display (display mode only) */
  if (!pargs.headless) {
    app.appsink = gst_bin_get_by_name(GST_BIN(app.pipeline), "videosink");
    app.appsrc = gst_bin_get_by_name(GST_BIN(app.pipeline), "display");
    if (app.appsrc) {
      GstCaps *caps = gst_caps_new_simple("video/x-raw",
          "format", G_TYPE_STRING, "RGBA",
          "width", G_TYPE_INT, app.srcWidth,
          "height", G_TYPE_INT, app.srcHeight,
          "framerate", GST_TYPE_FRACTION, 0, 1, NULL);
      g_object_set(app.appsrc, "caps", caps, NULL);
      gst_caps_unref(caps);
      app.dmabufAlloc = gst_dmabuf_allocator_new();
    }

    /* Forward EOS from appsink (background) to trigger video loop seek.
     * With sync=true on appsink, EOS arrives at real-time video duration.
     * Seek deferred to main loop via g_idle_add to avoid streaming thread deadlock. */
    if (app.appsink && app.appsrc) {
      GstPad *bgPad = gst_element_get_static_pad(app.appsink, "sink");
      if (bgPad) {
        gst_pad_add_probe(bgPad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
            [](GstPad *, GstPadProbeInfo *info, gpointer user_data) -> GstPadProbeReturn {
              if (GST_EVENT_TYPE(GST_PAD_PROBE_INFO_EVENT(info)) == GST_EVENT_EOS) {
                AppData *app = (AppData *)user_data;
                /* Defer seek to main loop — seeking from a streaming thread
                 * can deadlock. g_idle_add runs on the main thread. */
                g_idle_add([](gpointer data) -> gboolean {
                  AppData *a = (AppData *)data;
                  log_info("EOS — seeking to restart video\n");
                  gst_element_seek_simple(a->pipeline, GST_FORMAT_TIME,
                      (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT), 0);
                  /* Push a canvas buffer to appsrc immediately after seek to
                   * satisfy waylandsink preroll (async=false avoids this deadlock
                   * for initial startup but after seek we need to re-prime). */
                  if (a->appsrc && a->canvasFd[0] >= 0) {
                    int fd = dup(a->canvasFd[0]);
                    GstMemory *mem = gst_dmabuf_allocator_alloc(a->dmabufAlloc, fd, a->canvasSize);
                    GstBuffer *buf = gst_buffer_new();
                    gst_buffer_append_memory(buf, mem);
                    gst_app_src_push_buffer(GST_APP_SRC(a->appsrc), buf);
                  }
                  return G_SOURCE_REMOVE;
                }, app);
              }
              return GST_PAD_PROBE_OK;
            }, &app, NULL);
        gst_object_unref(bgPad);
      }
    }
  }

  /* Preprocessing timing probes */
  app.cameraadaptor = gst_bin_get_by_name(GST_BIN(app.pipeline), "preproc");
  if (app.cameraadaptor) {
    GstPad *sk = gst_element_get_static_pad(app.cameraadaptor, "sink");
    GstPad *sr = gst_element_get_static_pad(app.cameraadaptor, "src");
    if (sk) { gst_pad_add_probe(sk, GST_PAD_PROBE_TYPE_BUFFER, preprocStartProbe, &app, NULL); gst_object_unref(sk); }
    if (sr) { gst_pad_add_probe(sr, GST_PAD_PROBE_TYPE_BUFFER, preprocEndProbe, &app, NULL); gst_object_unref(sr); }
  }

  g_unix_signal_add(SIGINT, commonSigintHandler, &app.busCtx);

  gst_element_set_state(app.pipeline, GST_STATE_PLAYING);
  g_main_loop_run(app.loop);

  /* Cleanup */
  gst_element_set_state(app.pipeline, GST_STATE_NULL);
  printTiming(&app);

  for (auto &kv : app.frameTensorCache) hal_tensor_free(kv.second);
  for (int i = 0; i < AppData::NUM_CANVAS; i++) {
    if (app.canvasFd[i] >= 0) close(app.canvasFd[i]);
    if (app.canvas[i]) hal_tensor_free(app.canvas[i]);
  }
  if (app.decoder) hal_decoder_free(app.decoder);
  if (app.processor) hal_image_processor_free(app.processor);
  if (app.tensorFilter) gst_object_unref(app.tensorFilter);
  if (app.tensorSink) gst_object_unref(app.tensorSink);
  if (app.cameraadaptor) gst_object_unref(app.cameraadaptor);
  if (app.appsink) gst_object_unref(app.appsink);
  if (app.appsrc) gst_object_unref(app.appsrc);
  if (app.dmabufAlloc) gst_object_unref(app.dmabufAlloc);
  gst_object_unref(app.bus);
  gst_object_unref(app.pipeline);
  g_main_loop_unref(app.loop);

  return 0;
}
