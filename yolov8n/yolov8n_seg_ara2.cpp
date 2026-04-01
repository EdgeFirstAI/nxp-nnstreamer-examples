/**
 * Copyright 2026 EdgeFirst AI (Au-Zone Technologies)
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * YOLOv8-seg Instance Segmentation — Ara-2 NPU + EdgeFirst HAL
 *
 * Test binary for validating the segmentation decode pipeline:
 *   edgefirstcameraadaptor → ara2 tensor_filter → tensor_sink
 *
 * In the tensor_sink callback:
 *   1. Auto-configure HAL decoder from NNStreamer tensor caps
 *   2. Decode with hal_decoder_decode_proto() to get boxes + proto data
 *   3. Materialize masks with hal_image_processor_materialize_masks()
 *   4. Print detection boxes and mask dimensions
 *
 * Ara-2 YOLOv8-seg DVM output tensors (split format, 4 tensors):
 *   [0] boxes            — int16  [4, 8400]     (cx, cy, w, h)
 *   [1] scores           — uint8  [80, 8400]    (class confidences)
 *   [2] mask_coefficients — uint8  [32, 8400]    (per-detection mask coeffs)
 *   [3] protos           — uint8  [32, 160, 160] (prototype masks)
 *
 * Note: The actual tensor order and dtypes depend on the DVM compilation.
 * The auto-config logic detects tensor roles from their shapes, matching
 * the approach used by the edgefirstoverlay element.
 *
 * Usage:
 *   yolov8n_seg_ara2 -m /tmp/yolov8m-seg_640x640.dvm --platform imx8mp
 *   yolov8n_seg_ara2 -m /tmp/yolov8m-seg_640x640.dvm --platform imx8mp -v /tmp/video.mp4 -H -n 100
 */

#include <gst/gst.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/app/gstappsrc.h>
#include <glib.h>
#include <glib-unix.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
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
  int totalMasks;
  int framesWithDetections;
  int totalFrames;
  int maxFrames;

  GstElement *tensorFilter;
  GstElement *tensorSink;
  GstElement *cameraadaptor;
  LetterboxParams letterbox;

  bool headless;

  // HAL decoder + image processor
  hal_decoder *decoder;
  hal_image_processor *processor;

  // Display via appsrc: HAL renders masks onto canvas, pushed as DMA-BUF
  GstElement *appsrc;
  GstAllocator *dmabufAlloc;
  hal_tensor *canvas;         // DMA-backed RGBA at source resolution
  int srcWidth, srcHeight;

  // Tensor metadata from caps (filled on first frame)
  int tensor_count;
  hal_dtype tensor_dtypes[MAX_TENSORS];
  size_t tensor_shapes[MAX_TENSORS][8];   // NNStreamer squeezed shapes
  size_t tensor_ndims[MAX_TENSORS];
  size_t hal_shapes[MAX_TENSORS][8];      // HAL-convention shapes (batch prepended)
  size_t hal_ndims[MAX_TENSORS];
  bool decoder_configured;
};


/* ─── NNStreamer type parsing helpers (same as edgefirstoverlay) ─── */

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

/* Parse NNStreamer dim string "C:W:H:B" (innermost first) to shape[]
 * (row-major). Strips trailing :1 dimensions, then squeezes interior
 * unit dimensions (keeps min 2 dims). This handles Ara-2 DVM output
 * like "32:1:8400:1" → [8400, 32] instead of [8400, 1, 32]. */
static size_t parse_nnstreamer_dims(const char *dim_str, size_t *shape, size_t max_ndim)
{
  gchar **parts = g_strsplit(dim_str, ":", (gint)max_ndim + 1);
  size_t n = 0;
  while (parts[n] && n < max_ndim) n++;
  /* Strip trailing :1 dimensions (outermost batch dims) */
  while (n > 1 && g_strcmp0(parts[n - 1], "1") == 0) n--;
  /* NNStreamer: innermost first → reverse to row-major */
  size_t raw[8];
  for (size_t i = 0; i < n && i < 8; i++)
    raw[i] = (size_t)g_ascii_strtoull(parts[n - 1 - i], NULL, 10);
  g_strfreev(parts);
  /* Squeeze interior unit (=1) dimensions, keeping at least 2 dims */
  size_t out = 0;
  for (size_t i = 0; i < n; i++) {
    if (raw[i] != 1 || out == 0)
      shape[out++] = raw[i];
  }
  /* Ensure at least 2 dimensions */
  if (out < 2 && n >= 2) {
    shape[out++] = 1;
  }
  return out;
}


/* ─── Auto-configure HAL decoder from NNStreamer caps + quant meta ── */

static bool auto_config_decoder(AppData *app, GstCaps *caps, GstBuffer *buffer)
{
  GstStructure *s = gst_caps_get_structure(caps, 0);
  gint num_tensors = 0;
  const gchar *dims_str = NULL, *types_str = NULL;

  if (!gst_structure_get_int(s, "num_tensors", &num_tensors) || num_tensors <= 0)
    return false;

  dims_str  = gst_structure_get_string(s, "dimensions");
  types_str = gst_structure_get_string(s, "types");

  if (!dims_str || !types_str)
    return false;

  gchar **dim_parts  = g_strsplit(dims_str,  ",", num_tensors + 1);
  gchar **type_parts = g_strsplit(types_str, ",", num_tensors + 1);

  /* Parse shapes (squeeze interior unit dims) and dtypes */
  app->tensor_count = (num_tensors < MAX_TENSORS) ? num_tensors : MAX_TENSORS;

  /* After squeeze, shapes are in row-major order:
   *   protos: [H, W, C] (3D)  — e.g. [160, 160, 32]
   *   others: [num_boxes, features] (2D) — e.g. [8400, 80]
   */
  bool has_protos = false;
  size_t proto_channels = 0;

  for (int i = 0; i < app->tensor_count; i++) {
    app->tensor_ndims[i] = parse_nnstreamer_dims(dim_parts[i],
        app->tensor_shapes[i], 8);
    app->tensor_dtypes[i] = nnstreamer_type_to_hal(type_parts[i]);

    if (app->tensor_ndims[i] == 3) {
      has_protos = true;
      proto_channels = app->tensor_shapes[i][2]; /* [H, W, C] → C */
    }
  }

  /* Identify tensor roles (following ara2-rs/examples/yolov8.rs logic) */
  int protos_idx = -1, scores_idx = -1, boxes_idx = -1, coeffs_idx = -1;

  for (int i = 0; i < app->tensor_count; i++) {
    size_t ndim = app->tensor_ndims[i];
    if (ndim == 3) {
      protos_idx = i;  /* Only 3D tensor after squeeze = protos [H,W,C] */
    } else if (ndim == 2 && app->tensor_shapes[i][1] == 4) {
      boxes_idx = i;   /* [num_boxes, 4] */
    } else if (ndim == 2 && has_protos && app->tensor_shapes[i][1] == proto_channels) {
      coeffs_idx = i;  /* [num_boxes, 32] matching proto channels */
    } else if (scores_idx < 0) {
      scores_idx = i;  /* remaining 2D tensor = scores */
    }
  }

  /* Retry mask_coeff if proto appeared after it in the list */
  if (coeffs_idx < 0 && protos_idx >= 0) {
    for (int i = 0; i < app->tensor_count; i++) {
      if (i == protos_idx || i == boxes_idx || i == scores_idx) continue;
      if (app->tensor_ndims[i] == 2 && app->tensor_shapes[i][1] == proto_channels) {
        coeffs_idx = i;
        break;
      }
    }
  }

  log_info("Auto-config: %d tensors, proto_channels=%zu\n",
           app->tensor_count, proto_channels);
  for (int i = 0; i < app->tensor_count; i++) {
    const char *role = (i == protos_idx) ? "PROTOS" :
                       (i == boxes_idx)  ? "BOXES" :
                       (i == scores_idx) ? "SCORES" :
                       (i == coeffs_idx) ? "MASK_COEFF" : "?";
    char shape_str[128] = "";
    for (size_t j = 0; j < app->tensor_ndims[i]; j++) {
      char tmp[32];
      snprintf(tmp, sizeof(tmp), "%s%zu", j > 0 ? "x" : "", app->tensor_shapes[i][j]);
      strcat(shape_str, tmp);
    }
    log_info("  [%d] %-11s ndim=%zu shape=%s dtype=%d\n",
             i, role, app->tensor_ndims[i], shape_str, app->tensor_dtypes[i]);
  }

  g_strfreev(dim_parts);
  g_strfreev(type_parts);

  if (boxes_idx < 0 || scores_idx < 0) {
    log_error("Cannot identify boxes and scores tensors\n");
    return false;
  }

  /* Compute HAL-convention shapes with batch=1 prepended.
   * 2D [num_boxes, features] → [1, features, num_boxes]
   * 3D protos [H, W, C] → [1, C, H, W] */
  for (int i = 0; i < app->tensor_count; i++) {
    size_t *ns = app->tensor_shapes[i];
    if (i == protos_idx) {
      /* [H, W, C] → [1, C, H, W] */
      app->hal_shapes[i][0] = 1;
      app->hal_shapes[i][1] = ns[2]; /* C */
      app->hal_shapes[i][2] = ns[0]; /* H */
      app->hal_shapes[i][3] = ns[1]; /* W */
      app->hal_ndims[i] = 4;
    } else {
      /* [num_boxes, features] → [1, features, num_boxes] */
      app->hal_shapes[i][0] = 1;
      app->hal_shapes[i][1] = ns[1]; /* features */
      app->hal_shapes[i][2] = ns[0]; /* num_boxes */
      app->hal_ndims[i] = 3;
    }
  }

  /* Extract quant params from GstNnsTensorQuantMeta (attached by tensor_filter).
   * For box outputs, divide scale by input_dim to normalize to [0,1] coords. */
  QuantParams qp[MAX_TENSORS] = {};
  for (int i = 0; i < app->tensor_count; i++) {
    if (!extractQuantParams(buffer, i, qp[i])) {
      qp[i].scale = 1.0;
      qp[i].zeroPoint = 0;
      log_info("  [%d] no quant meta, using scale=1.0 zp=0\n", i);
    } else {
      log_info("  [%d] quant: scale=%g zp=%" G_GINT64_FORMAT "\n",
               i, qp[i].scale, qp[i].zeroPoint);
    }
  }

  /* Box quant: divide by input_dim for normalized [0,1] coordinates
   * (same as ara2-rs: info.quant.qn / input_dim) */
  if (boxes_idx >= 0) {
    qp[boxes_idx].scale /= (double)MODEL_INPUT_SIZE;
    log_info("  boxes adj. scale=%g (÷%d)\n", qp[boxes_idx].scale, MODEL_INPUT_SIZE);
  }

  /* Build decoder with programmatic API.
   *
   * Shapes must match ONNX convention with batch=1 prepended:
   *   scores:     [1, num_classes, num_boxes]
   *   boxes:      [1, 4, num_boxes]
   *   mask_coeff: [1, num_protos, num_boxes]
   *   protos:     [1, C, H, W]              ← 4D! */
  hal_decoder_params *params = hal_decoder_params_new();
  hal_decoder_params_set_score_threshold(params, CONF_THRESHOLD);
  hal_decoder_params_set_iou_threshold(params, NMS_IOU_THRESHOLD);
  hal_decoder_params_set_nms(params, HAL_NMS_CLASS_AGNOSTIC);

  /* Helper: add a split output with shape [1, features, num_boxes] */
  auto add_split_output = [&](int tensor_idx, HalOutputType type,
                              HalDimName dim1_name) -> int {
    size_t num_boxes = app->tensor_shapes[tensor_idx][0];
    size_t features  = app->tensor_shapes[tensor_idx][1];
    size_t shape[3] = {1, features, num_boxes};
    HalDimName dims[3] = {HAL_DIM_NAME_BATCH, dim1_name, HAL_DIM_NAME_NUM_BOXES};
    int idx = hal_decoder_params_add_output(params, type,
        HAL_DECODER_TYPE_ULTRALYTICS, shape, dims, 3);
    if (idx >= 0)
      hal_decoder_params_output_set_quantization(params, idx,
          (float)qp[tensor_idx].scale, (int)qp[tensor_idx].zeroPoint);
    if (type == HAL_OUTPUT_TYPE_BOXES && idx >= 0)
      hal_decoder_params_output_set_normalized(params, idx, 1);
    return idx;
  };

  /* Scores: [1, num_classes, num_boxes] */
  int si = add_split_output(scores_idx, HAL_OUTPUT_TYPE_SCORES, HAL_DIM_NAME_NUM_CLASSES);
  log_info("  added SCORES idx=%d\n", si);

  /* Boxes: [1, 4, num_boxes] */
  int bi = add_split_output(boxes_idx, HAL_OUTPUT_TYPE_BOXES, HAL_DIM_NAME_BOX_COORDS);
  log_info("  added BOXES idx=%d\n", bi);

  /* Mask coefficients: [1, num_protos, num_boxes] */
  if (coeffs_idx >= 0) {
    int mi = add_split_output(coeffs_idx, HAL_OUTPUT_TYPE_MASK_COEFFICIENTS, HAL_DIM_NAME_NUM_PROTOS);
    log_info("  added MASK_COEFF idx=%d\n", mi);
  }

  /* Protos: [1, C, H, W] — 4D with batch dim */
  if (protos_idx >= 0) {
    size_t H = app->tensor_shapes[protos_idx][0];
    size_t W = app->tensor_shapes[protos_idx][1];
    size_t C = app->tensor_shapes[protos_idx][2];
    size_t shape[4] = {1, C, H, W};
    HalDimName dims[4] = {HAL_DIM_NAME_BATCH, HAL_DIM_NAME_NUM_PROTOS,
                          HAL_DIM_NAME_HEIGHT, HAL_DIM_NAME_WIDTH};
    int pi = hal_decoder_params_add_output(params, HAL_OUTPUT_TYPE_PROTOS,
        HAL_DECODER_TYPE_ULTRALYTICS, shape, dims, 4);
    if (pi >= 0)
      hal_decoder_params_output_set_quantization(params, pi,
          (float)qp[protos_idx].scale, (int)qp[protos_idx].zeroPoint);
    log_info("  added PROTOS idx=%d shape=[1,%zu,%zu,%zu]\n", pi, C, H, W);
  }

  app->decoder = hal_decoder_new(params);
  hal_decoder_params_free(params);

  if (!app->decoder) {
    log_error("HAL decoder creation failed (errno=%d: %s)\n", errno, strerror(errno));
    return false;
  }

  int normalized = hal_decoder_normalized_boxes(app->decoder);
  char *model_type = hal_decoder_model_type(app->decoder);
  log_info("HAL decoder created: type=%s, normalized=%d\n",
           model_type ? model_type : "unknown", normalized);
  free(model_type);

  app->decoder_configured = true;
  return true;
}


/* ─── Preprocessing timing probes ────────────────────────────────── */

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


/* ─── Timing report ──────────────────────────────────────────────── */

static void printTiming(void *userData)
{
  AppData *app = (AppData *)userData;

  printf("\n");
  printf("==============================================================================\n");
  printf("  YOLOV8-SEG — ARA-2 NPU + EDGEFIRST HAL SEGMENTATION DECODE\n");
  printf("==============================================================================\n");

  printf("\n  PREPROCESSING (edgefirstcameraadaptor)\n");
  printf("  --------------------------------------------------------------------------\n");
  if (app->preproc.count > 0)
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d frames]\n",
           app->preproc.avg(), app->preproc.minMs, app->preproc.maxMs, app->preproc.count);

  printf("\n  INFERENCE (Ara-2 NPU)\n");
  printf("  --------------------------------------------------------------------------\n");
  if (app->inference.count > 0)
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d samples]\n",
           app->inference.avg(), app->inference.minMs, app->inference.maxMs, app->inference.count);

  printf("\n  POST-PROCESSING (HAL decode_proto + materialize_masks)\n");
  printf("  --------------------------------------------------------------------------\n");
  if (app->postproc.count > 0)
    printf("     Average: %7.3f ms  |  Min: %7.3f ms  |  Max: %7.3f ms  [%d frames]\n",
           app->postproc.avg(), app->postproc.minMs, app->postproc.maxMs, app->postproc.count);

  printf("\n  END-TO-END\n");
  printf("  --------------------------------------------------------------------------\n");
  if (app->throughput.metric.count > 0) {
    double avgMs = app->throughput.metric.avg();
    printf("     Average: %7.3f ms (%5.1f FPS)  |  Frames: %d\n",
           avgMs, 1000.0 / avgMs, app->throughput.metric.count);
  }

  printf("\n  DETECTION + SEGMENTATION STATISTICS\n");
  printf("  --------------------------------------------------------------------------\n");
  printf("     Post-NMS detections:     %6d  (avg %.1f/frame)\n",
         app->totalDetections,
         app->framesWithDetections > 0
             ? (double)app->totalDetections / app->framesWithDetections : 0.0);
  printf("     Materialized masks:      %6d\n", app->totalMasks);
  printf("     Frames with detections:  %6d / %d\n",
         app->framesWithDetections, app->totalFrames);
  printf("\n==============================================================================\n");
}


/* ─── tensor_sink new-data callback ──────────────────────────────── */

static void newDataCallback(GstElement *element, GstBuffer *buffer, gpointer user_data)
{
  struct timeval startTime, endTime;
  gettimeofday(&startTime, NULL);

  AppData *app = (AppData *)user_data;

  /* Query letterbox from cameraadaptor on first frame */
  if (app->throughput.firstFrame && app->cameraadaptor) {
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

  /* Frame-to-frame interval */
  app->throughput.tick(startTime);

  /* Inference latency */
  queryInferenceLatency(app->tensorFilter, app->inference);

  /* Auto-configure decoder on first frame using caps */
  if (!app->decoder_configured) {
    GstPad *sinkpad = gst_element_get_static_pad(element, "sink");
    if (sinkpad) {
      GstCaps *caps = gst_pad_get_current_caps(sinkpad);
      if (caps) {
        gchar *caps_str = gst_caps_to_string(caps);
        log_info("Tensor caps: %s\n", caps_str);
        g_free(caps_str);

        if (!auto_config_decoder(app, caps, buffer)) {
          log_error("Failed to auto-configure decoder from caps\n");
          gst_caps_unref(caps);
          gst_object_unref(sinkpad);
          g_main_loop_quit(app->loop);
          return;
        }
        gst_caps_unref(caps);
      }
      gst_object_unref(sinkpad);
    }
  }

  if (!app->decoder) {
    log_error("Decoder not ready\n");
    return;
  }

  /* Validate buffer */
  if (!GST_IS_BUFFER(buffer)) { log_error("Invalid buffer\n"); return; }
  guint n_mem = gst_buffer_n_memory(buffer);
  if ((int)n_mem != app->tensor_count) {
    log_error("Expected %d tensors, got %u\n", app->tensor_count, n_mem);
    return;
  }

  /* Wrap output tensors as HAL tensors */
  hal_tensor *hal_outputs[MAX_TENSORS] = {};
  GstMapInfo out_maps[MAX_TENSORS] = {};
  gboolean out_mapped[MAX_TENSORS] = {};
  hal_detect_box_list *boxes = NULL;
  hal_proto_data *proto = NULL;
  hal_segmentation_list *segs = NULL;

  for (int j = 0; j < app->tensor_count; j++) {
    GstMemory *mem = gst_buffer_peek_memory(buffer, j);

    /* Use HAL-convention shapes (batch prepended, features before boxes)
     * to match the decoder config from auto_config_decoder. */
    if (gst_is_dmabuf_memory(mem)) {
      int fd = gst_dmabuf_memory_get_fd(mem);
      hal_outputs[j] = hal_tensor_from_fd(
          app->tensor_dtypes[j], dup(fd),
          app->hal_shapes[j], app->hal_ndims[j], NULL);
    } else {
      if (!gst_memory_map(mem, &out_maps[j], GST_MAP_READ)) {
        log_error("Cannot map output tensor %d\n", j);
        goto cleanup;
      }
      out_mapped[j] = TRUE;

      hal_outputs[j] = hal_tensor_new(
          app->tensor_dtypes[j], app->hal_shapes[j],
          app->hal_ndims[j], HAL_TENSOR_MEMORY_MEM, NULL);

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
      log_error("Failed to create HAL tensor for output %d\n", j);
      goto cleanup;
    }
  }

  /* Step 1: Decode with proto data */
  proto = hal_decoder_decode_proto(
      app->decoder,
      (const hal_tensor *const *)hal_outputs, app->tensor_count,
      &boxes);

  if (!boxes) {
    static bool first_error = true;
    if (first_error) {
      log_error("hal_decoder_decode_proto failed (errno=%d: %s)\n",
                errno, strerror(errno));
      first_error = false;
    }
    goto cleanup;
  }

  /* Step 2: Materialize masks if proto data and processor are available */
  if (proto && app->processor) {
    /* Build letterbox array [x0, y0, x1, y1] in normalized coordinates */
    float lb[4] = {0};
    if (app->letterbox.scale > 0) {
      lb[0] = (float)app->letterbox.padX / MODEL_INPUT_SIZE;
      lb[1] = (float)app->letterbox.padY / MODEL_INPUT_SIZE;
      lb[2] = 1.0f - lb[0];
      lb[3] = 1.0f - lb[1];
    }

    segs = hal_image_processor_materialize_masks(
        app->processor, boxes, proto,
        app->letterbox.scale > 0 ? lb : NULL);
  }

  /* Process results */
  {
    size_t num_dets = hal_detect_box_list_len(boxes);
    size_t num_segs = segs ? hal_segmentation_list_len(segs) : 0;
    app->totalFrames++;

    if (num_dets > 0) {
      app->framesWithDetections++;
      app->totalDetections += num_dets;
      app->totalMasks += num_segs;

      /* Print first frame detections with mask info */
      static bool first_print = true;
      if (first_print) {
        log_info("Detected objects (%zu detections, %zu masks):\n", num_dets, num_segs);
        for (size_t d = 0; d < num_dets; d++) {
          hal_detect_box box;
          if (hal_detect_box_list_get(boxes, d, &box) == 0) {
            const char *name = (box.label >= 0 && box.label < NUM_CLASSES)
                ? cocoClassNames[box.label] : "?";

            if (segs && d < num_segs) {
              size_t mask_h = 0, mask_w = 0;
              const uint8_t *mask = hal_segmentation_list_get_mask(segs, d, &mask_h, &mask_w);
              /* Count non-zero mask pixels */
              size_t mask_pixels = 0;
              if (mask) {
                for (size_t p = 0; p < mask_h * mask_w; p++) {
                  if (mask[p] > 127) mask_pixels++;
                }
              }
              log_info("  [%zu] %-14s %3.0f%%  box=(%.1f,%.1f,%.1f,%.1f)  mask=%zux%zu (%zu px)\n",
                       d, name, box.score * 100.0f,
                       box.xmin, box.ymin, box.xmax, box.ymax,
                       mask_w, mask_h, mask_pixels);
            } else {
              log_info("  [%zu] %-14s %3.0f%%  box=(%.1f,%.1f,%.1f,%.1f)  NO MASK\n",
                       d, name, box.score * 100.0f,
                       box.xmin, box.ymin, box.xmax, box.ymax);
            }
          }
        }
        log_info("\n");

        if (!proto)
          log_info("NOTE: decode_proto returned NULL proto — model may be detection-only\n\n");

        first_print = false;
      }
    }

    /* Frame limit */
    if (app->maxFrames > 0 && app->totalFrames >= app->maxFrames) {
      gettimeofday(&endTime, NULL);
      app->postproc.record(timeDiffMs(startTime, endTime));
      goto cleanup_quit;
    }

    /* Step 3: Draw masks onto canvas and push to appsrc for display */
    if (app->appsrc && app->processor && segs) {
      /* Lazy-create DMA-backed RGBA canvas at source resolution */
      if (!app->canvas) {
        app->canvas = hal_image_processor_create_image(
            app->processor, app->srcWidth, app->srcHeight,
            HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
        if (app->canvas)
          log_info("Display canvas: %dx%d RGBA (DMA-BUF)\n",
                   app->srcWidth, app->srcHeight);
        else
          log_error("Failed to create display canvas\n");
      }

      if (app->canvas) {
        /* Letterbox in normalized coords for draw_decoded_masks */
        float lb[4] = {0};
        if (app->letterbox.scale > 0) {
          lb[0] = (float)app->letterbox.padX / MODEL_INPUT_SIZE;
          lb[1] = (float)app->letterbox.padY / MODEL_INPUT_SIZE;
          lb[2] = 1.0f - lb[0];
          lb[3] = 1.0f - lb[1];
        }

        int draw_ret = hal_image_processor_draw_decoded_masks(
            app->processor, app->canvas,
            boxes, segs,
            NULL,  /* no background for now */
            0.5f,  /* opacity */
            app->letterbox.scale > 0 ? lb : NULL,
            HAL_COLOR_MODE_INSTANCE);

        if (draw_ret == 0) {
          /* Wrap canvas DMA-BUF fd as GstBuffer and push to appsrc */
          int fd = hal_tensor_clone_fd(app->canvas);
          if (fd >= 0) {
            size_t buf_size = hal_tensor_size(app->canvas);
            GstMemory *mem = gst_dmabuf_allocator_alloc(
                app->dmabufAlloc, fd, buf_size);
            GstBuffer *outBuf = gst_buffer_new();
            gst_buffer_append_memory(outBuf, mem);

            GstFlowReturn flow = gst_app_src_push_buffer(
                GST_APP_SRC(app->appsrc), outBuf);
            if (flow != GST_FLOW_OK) {
              static bool first = true;
              if (first) {
                log_error("appsrc push failed: %s\n",
                          gst_flow_get_name(flow));
                first = false;
              }
            }
          }
        } else {
          static bool first = true;
          if (first) {
            log_error("draw_decoded_masks failed (ret=%d)\n", draw_ret);
            first = false;
          }
        }
      }
    }
  }

cleanup:
  if (segs) hal_segmentation_list_free(segs);
  if (proto) hal_proto_data_free(proto);
  if (boxes) hal_detect_box_list_free(boxes);
  for (int j = 0; j < app->tensor_count; j++) {
    if (hal_outputs[j]) hal_tensor_free(hal_outputs[j]);
  }
  for (int j = 0; j < app->tensor_count; j++) {
    if (out_mapped[j])
      gst_memory_unmap(gst_buffer_peek_memory(buffer, j), &out_maps[j]);
  }

  gettimeofday(&endTime, NULL);
  app->postproc.record(timeDiffMs(startTime, endTime));
  return;

cleanup_quit:
  if (segs) hal_segmentation_list_free(segs);
  if (proto) hal_proto_data_free(proto);
  if (boxes) hal_detect_box_list_free(boxes);
  for (int j = 0; j < app->tensor_count; j++) {
    if (hal_outputs[j]) hal_tensor_free(hal_outputs[j]);
  }
  for (int j = 0; j < app->tensor_count; j++) {
    if (out_mapped[j])
      gst_memory_unmap(gst_buffer_peek_memory(buffer, j), &out_maps[j]);
  }
  g_main_loop_quit(app->loop);
}


/* ─── Main ───────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
  ParsedArgs pargs;
  pargs.camera = "";

  uint32_t flags = ARG_MODEL | ARG_CAMERA | ARG_VIDEO | ARG_IMAGE |
                   ARG_HEADLESS | ARG_NUM_FRAMES | ARG_PLATFORM;

  int ret = parseArgs(argc, argv, flags,
      "YOLOv8-seg Instance Segmentation — Ara-2 NPU + EdgeFirst HAL", pargs);
  if (ret != 0) return ret > 0 ? 0 : 1;

  if (pargs.model.empty()) {
    log_error("Provide model path with -m\n");
    return 1;
  }

  /* Parse platform */
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

  if (pargs.camera.empty() && plat.defaultCamera)
    pargs.camera = plat.defaultCamera;

  if (platform == PLATFORM_IMX95)
    setupImx95Environment(false);

  gst_init(&argc, &argv);

  log_info("YOLOv8-seg for Ara-2 NPU — %s\n", plat.name);
  log_info("Model: %s\n", pargs.model.c_str());
  if (!pargs.image.empty())
    log_info("Input: image (%s)\n", pargs.image.c_str());
  else if (!pargs.video.empty())
    log_info("Input: video (%s)\n", pargs.video.c_str());
  else
    log_info("Input: camera (%s)\n", pargs.camera.empty() ? "libcamerasrc" : pargs.camera.c_str());
  if (pargs.numFrames > 0)
    log_info("Frames: %d\n", pargs.numFrames);
  log_info("Mode: %s\n", pargs.headless ? "headless" : "display");

  /* Build source element */
  InputSource srcType = determineInputSource(pargs, plat.usesLibcamerasrc);
  char *srcStr = buildSourceElement(srcType, pargs);

  /* Build NN processing pipeline — same as yolov8n_ara2 */
  char *nnBranch = g_strdup_printf(
      "queue name=thread-nn leaky=2 max-size-buffers=2 ! "
      "edgefirstcameraadaptor name=preproc model-width=%d model-height=%d "
      "model-dtype=int8 model-layout=chw letterbox=true ! "
      "tensor_filter name=tfilter framework=ara2 model=%s "
      "custom=EnableStats:true latency=1 ! "
      "tensor_sink name=inferenceOutput",
      MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
      pargs.model.c_str());

  /* Build full pipeline.
   * Display mode: appsrc pushes HAL-rendered RGBA frames to waylandsink.
   * The appsrc is a separate bin linked via gst_parse_launch; we push
   * DMA-BUF-backed buffers from the tensor_sink callback after draw_decoded_masks. */
  char *pipelineStr;
  if (pargs.headless) {
    pipelineStr = g_strdup_printf("%s ! %s", srcStr, nnBranch);
  } else {
    pipelineStr = g_strdup_printf(
        "%s ! %s "
        "appsrc name=display format=3 is-live=true do-timestamp=true "
        "max-buffers=2 block=false ! "
        "waylandsink sync=false async=false",
        srcStr, nnBranch);
  }
  g_free(srcStr);
  g_free(nnBranch);

  log_info("Pipeline: %s\n\n", pipelineStr);

  AppData app = {};
  app.maxFrames = pargs.numFrames;
  app.headless = pargs.headless;
  app.preproc.reset();
  app.inference.reset();
  app.postproc.reset();
  app.throughput.reset();

  /* Create image processor early (before pipeline) with OpenGL backend
   * for GPU-accelerated mask materialization and overlay rendering. */
  log_info("Creating HAL image processor (OpenGL backend)...\n");
  fflush(stdout);
  app.processor = hal_image_processor_new_with_backend(HAL_COMPUTE_BACKEND_OPENGL);
  log_info("hal_image_processor_new_with_backend returned %p\n", (void *)app.processor);
  fflush(stdout);
  if (!app.processor) {
    log_error("HAL image processor creation failed (errno=%d: %s)\n", errno, strerror(errno));
    log_info("Continuing without mask materialization — decode_proto only\n");
  } else {
    log_info("HAL image processor ready\n");
  }

  bool startedOnce = false;
  app.busCtx.playing = &app.playing;
  app.busCtx.startedOnce = pargs.headless ? NULL : &startedOnce;
  app.busCtx.videoLoop = !pargs.video.empty() && pargs.image.empty();
  app.busCtx.videoRate = 1.0;
  app.busCtx.printTiming = NULL;
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
  app.tensorSink = gst_bin_get_by_name(GST_BIN(app.pipeline), "inferenceOutput");
  g_signal_connect(app.tensorSink, "new-data", G_CALLBACK(newDataCallback), &app);

  /* appsrc for display (non-headless only) */
  if (!pargs.headless) {
    app.appsrc = gst_bin_get_by_name(GST_BIN(app.pipeline), "display");
    if (app.appsrc) {
      app.srcWidth = SOURCE_WIDTH;
      app.srcHeight = SOURCE_HEIGHT;
      GstCaps *appCaps = gst_caps_new_simple("video/x-raw",
          "format", G_TYPE_STRING, "RGBA",
          "width", G_TYPE_INT, app.srcWidth,
          "height", G_TYPE_INT, app.srcHeight,
          "framerate", GST_TYPE_FRACTION, 0, 1,
          NULL);
      /* Advertise DMA-BUF capability */
      gst_caps_set_features(appCaps, 0,
          gst_caps_features_new(GST_CAPS_FEATURE_MEMORY_DMABUF, NULL));
      g_object_set(app.appsrc, "caps", appCaps, NULL);
      gst_caps_unref(appCaps);
      app.dmabufAlloc = gst_dmabuf_allocator_new();
      log_info("Display: appsrc configured for %dx%d RGBA DMA-BUF\n",
               app.srcWidth, app.srcHeight);
    }
  }

  /* tensor_filter for latency query */
  app.tensorFilter = gst_bin_get_by_name(GST_BIN(app.pipeline), "tfilter");

  /* Preprocessing timing probes */
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

  g_unix_signal_add(SIGINT, commonSigintHandler, &app.busCtx);

  /* Run */
  gst_element_set_state(app.pipeline, GST_STATE_PLAYING);
  g_main_loop_run(app.loop);

  /* Cleanup */
  gst_element_set_state(app.pipeline, GST_STATE_NULL);
  printTiming(&app);
  if (app.tensorFilter) gst_object_unref(app.tensorFilter);
  if (app.tensorSink) gst_object_unref(app.tensorSink);
  if (app.cameraadaptor) gst_object_unref(app.cameraadaptor);
  gst_object_unref(app.bus);
  gst_object_unref(app.pipeline);
  g_main_loop_unref(app.loop);
  if (app.canvas) hal_tensor_free(app.canvas);
  if (app.decoder) hal_decoder_free(app.decoder);
  if (app.processor) hal_image_processor_free(app.processor);
  if (app.appsrc) gst_object_unref(app.appsrc);
  if (app.dmabufAlloc) gst_object_unref(app.dmabufAlloc);

  return 0;
}
