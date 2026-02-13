/**
 * Copyright 2025 NXP
 * SPDX-License-Identifier: BSD-3-Clause 
 */ 

#include <gst/gst.h>
#include <glib.h>
#include <glib-unix.h>
#include <stdio.h>
#include <cairo.h>
#include <omp.h>
#include <vector>
#include <cstdint>
#include <string>
#include <sys/time.h>
#include <algorithm>
#include <getopt.h>
#include <iomanip>
#include <iostream>

#include "logging.hpp"
#include "nms.hpp"


#define PROFILING


// Input image dimension
#define IMAGE_WIDTH  640
#define IMAGE_HEIGHT 480


// Model parameters can be found using Netron.app website, or eIQ ToolKit
#define MODEL_INPUT_SIZE    320       // dimension of the input image to be detected
#define MODEL_OUTPUT_WIDTH  84        // width of the output layer tensor
#define NUM_TOTAL_BOXES     2100      // height of the output layer tensor
#define NUM_COORDINATES     4         // number of bounding boxes attributes (cx, cy, h, w)
#define ZERO_POINT          128       // quantization parameter of the output layer used for dequantization
#define SCALE_FACTOR        0.004194  // quantization parameter of the output layer used for dequantization


// Post-processing parameters
#define CONF_THRESHOLD      0.25
#define QUANTIZED_THRESHOLD (CONF_THRESHOLD/(SCALE_FACTOR - ZERO_POINT))


/** @brief Parameters to calculate post-processing durations */
typedef struct {
  double sinkTotalDuration;
  int sinkIter;
  double cairoTotalDuration;
  int cairoIter;
} PostProcessDuration;


/** @brief Data structure for app. */
typedef struct {
  GstElement *gstPipeline;
  GMainLoop *loop;
  GstBus *bus;
  gboolean playing;
  std::vector<DetectedObject> results;
  std::vector<std::string> className;
  PostProcessDuration ppDuration;
} AppData;


class YOLOv8 {
  private:
    char *strPipeline;
    AppData gApp;
    std::string model;
    std::string image;

  public:
    YOLOv8(const std::string &model, const std::string &image) : model(model), image(image)
    {
      /* init application parameters */
      gApp.gstPipeline = nullptr;
      gApp.loop = nullptr;
      gApp.bus = nullptr;
      gApp.playing = false;
      gApp.className  = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
        "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
        "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
        "frisbee","skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
      };
      gApp.ppDuration = {0};

      /* init GStreamer */
      gst_init(nullptr, nullptr);

      /* init pipeline */
      this->strPipeline = g_strdup_printf("filesrc location=%s ! jpegdec ! imxvideoconvert_g2d ! imagefreeze ! "
          "video/x-raw,width=%d,height=%d ! "
          "tee name=t "
          "t. ! queue name=thread-nn leaky=2 max-size-buffers=2 ! "
          "imxvideoconvert_g2d ! video/x-raw,width=%d,height=%d,format=RGBA ! "
          "videoconvert ! video/x-raw,format=RGB ! "
          "tensor_converter ! "
          "tensor_transform mode=arithmetic option=typecast:int16,add:-128 ! "
          "tensor_transform mode=typecast option=int8 ! "
          "tensor_filter framework=tensorflow-lite model=%s custom=Delegate:External,ExtDelegateLib:libneutron_delegate.so ! "
          "tensor_sink name=inferenceOutput "
          "t. ! queue name=thread-img max-size-buffers=2 ! "
          "imxvideoconvert_g2d ! "
          "cairooverlay name=cairo ! "
          "waylandsink ",
          this->image.c_str(),
          IMAGE_WIDTH,
          IMAGE_HEIGHT,
          MODEL_INPUT_SIZE,
          MODEL_INPUT_SIZE,
          this->model.c_str());

      log_info("%s\n\n", this->strPipeline);
    }

    void run()
    {
      log_info("start app...\n");

      /* main loop */
      this->gApp.loop = g_main_loop_new(NULL, FALSE);

      /* create GStreamer pipeline */
      this->gApp.gstPipeline = gst_parse_launch(this->strPipeline, NULL);

      /* bus and message callback */
      gApp.bus = gst_element_get_bus(gApp.gstPipeline);
      gst_bus_add_signal_watch(gApp.bus);
      g_signal_connect(gApp.bus, "message", G_CALLBACK (busCallback), &gApp);

      GstElement *tensorsink = gst_bin_get_by_name(GST_BIN(this->gApp.gstPipeline), "inferenceOutput");
      g_signal_connect(tensorsink, "new-data", G_CALLBACK(newDataCallback), &this->gApp);
    
      GstElement *cairo = gst_bin_get_by_name(GST_BIN(this->gApp.gstPipeline), "cairo");
      g_signal_connect(cairo, "draw", G_CALLBACK(drawCallback), &this->gApp);

      /* shutdowm pipeline with SIGINT signal */
      g_unix_signal_add(SIGINT, sigintSignalHandler, &this->gApp);

      /* start pipeline */
      gst_element_set_state(this->gApp.gstPipeline, GST_STATE_PLAYING);

      /* run main loop, quit when received eos or error message */
      g_main_loop_run(this->gApp.loop);

      /* end pipeline */
      gst_element_set_state(this->gApp.gstPipeline, GST_STATE_NULL);

      log_info("close app...\n");

      freeData();
    }

    static gboolean sigintSignalHandler(gpointer user_data)
    {
      AppData* gApp = (AppData *) user_data;

      log_info("SIGINT signal detected.\n");
      log_info("closing the main loop.\n");
      g_print("model output post-process average duration: %.2f ms\n",
              gApp->ppDuration.sinkTotalDuration/gApp->ppDuration.sinkIter);
      g_print("boxes rendering average duration: %.2f ms\n",
              gApp->ppDuration.cairoTotalDuration/gApp->ppDuration.cairoIter);
      g_main_loop_quit(gApp->loop);

      return true;
    }

    void freeData()
    {
      if (this->gApp.loop) {
        g_main_loop_unref(this->gApp.loop);
        this->gApp.loop = NULL;
      }
      if (this->gApp.gstPipeline) {
        gst_object_unref(this->gApp.gstPipeline);
        this->gApp.gstPipeline = NULL;
      }
    }

    /**
     * @brief Callback to manage GStreamer bus.
     */
    static void busCallback(GstBus *bus,
                            GstMessage *message,
                            gpointer user_data)
    {
      AppData *gApp = (AppData *) user_data;
      switch (GST_MESSAGE_TYPE (message))
      {
        case GST_MESSAGE_ERROR: {
          GError *err;
          gchar *debugInfo;

          gst_message_parse_error(message, &err, &debugInfo);
          log_error("Error received from element %s: %s.\n", GST_OBJECT_NAME(message->src), err->message);
          log_debug("Debugging information: %s.\n", debugInfo ? debugInfo : "none");
          g_error_free(err);
          g_free(debugInfo);
          log_info("Closing the main loop.\n");
          g_main_loop_quit(gApp->loop);
          break;
        }
        case GST_MESSAGE_EOS: {
          log_info("End-Of-Stream reached.\n");
          log_info("Closing the main loop.\n");
          g_main_loop_quit(gApp->loop);
          break;
        }
        case GST_MESSAGE_STATE_CHANGED: {
          GstState oldState, newState, pendingState;
          gst_message_parse_state_changed(message, &oldState, &newState, &pendingState);
          if (GST_MESSAGE_SRC(message) == GST_OBJECT(gApp->gstPipeline)) {
            log_info("Pipeline state changed from %s to %s.\n",
            gst_element_state_get_name(oldState), gst_element_state_get_name(newState));

            gApp->playing = (newState == GST_STATE_PLAYING);
          }
        }
        default:
          break;
      }
    }

    static void newDataCallback(GstElement *element,
                                GstBuffer *buffer,
                                gpointer user_data)
    {
      #ifdef PROFILING
      struct timeval preprocessStart, preprocessEnd;
      gettimeofday(&preprocessStart, NULL);
      #endif

      AppData *gApp = (AppData *) user_data;

      /* check number of tensor */
      if (!GST_IS_BUFFER (buffer)) {
        log_error("Received invalid buffer\n");
        exit(-1);
      }
      uint mem_blocks = gst_buffer_n_memory(buffer);
      if (mem_blocks != 1) {
        log_error("Number of tensors invalid\n");
        exit(-1);
      }

      /* get output tensor size and data */
      int8_t *outputTensor;
      GstMapInfo info;
      GstMemory *memKpts = gst_buffer_peek_memory(buffer, 0);
      bool result = gst_memory_map(memKpts, &info, GST_MAP_READ);
      if (result) {
        outputTensor = reinterpret_cast<int8_t*>(info.data);
        gst_memory_unmap(memKpts, &info);
      } else {
        gst_memory_unmap(memKpts, &info);
        log_error("Can't access buffer in memory\n");
        exit(-1);
      }

      std::vector <DetectedObject> output;
      for (int bIdx = 0; bIdx < NUM_TOTAL_BOXES; ++bIdx) {
        int maxClassConfVal = -128;
        int maxClassIdx = -1;
        for (int cIdx = NUM_COORDINATES; cIdx < MODEL_OUTPUT_WIDTH; cIdx++) {
          if (outputTensor[cIdx * NUM_TOTAL_BOXES + bIdx] > maxClassConfVal) {
            maxClassConfVal = outputTensor[cIdx * NUM_TOTAL_BOXES + bIdx];
            maxClassIdx = cIdx;
          }
        }
        if (maxClassConfVal > QUANTIZED_THRESHOLD) {
          DetectedObject object;
          float cx, cy, w, h;
          cx = (outputTensor[bIdx] + ZERO_POINT) * SCALE_FACTOR;
          cy = (outputTensor[NUM_TOTAL_BOXES + bIdx] + ZERO_POINT) * SCALE_FACTOR;
          w = (outputTensor[2 * NUM_TOTAL_BOXES + bIdx] + ZERO_POINT) * SCALE_FACTOR;
          h = (outputTensor[3 * NUM_TOTAL_BOXES + bIdx] + ZERO_POINT) * SCALE_FACTOR;
          cx *= static_cast<float>(IMAGE_WIDTH);
          cy *= static_cast<float>(IMAGE_HEIGHT);
          w *= static_cast<float>(IMAGE_WIDTH);
          h *= static_cast<float>(IMAGE_HEIGHT);

          object.x = static_cast<int>(std::max(0.f, (cx - w / 2.f)));
          object.y = static_cast<int>(std::max(0.f, (cy - h / 2.f)));
          object.width = static_cast<int>(std::min(static_cast<float>(IMAGE_WIDTH), w));
          object.height = static_cast<int>(std::min(static_cast<float>(IMAGE_HEIGHT), h));

          object.classId = maxClassIdx - NUM_COORDINATES;
          object.valid = true;
          output.push_back(object);
        }
      }

      gApp->results = output;
      output.clear();
      nms(gApp->results);

      #ifdef PROFILING
      gettimeofday(&preprocessEnd, NULL);
      gApp->ppDuration.sinkIter += 1;
      gApp->ppDuration.sinkTotalDuration += 1000.0 * (preprocessEnd.tv_sec-preprocessStart.tv_sec)
                                         + (preprocessEnd.tv_usec-preprocessStart.tv_usec) / 1000.0;
      #endif
    }

    static void drawCallback(GstElement *overlay,
                             cairo_t *cr,
                             guint64 timestamp,
                             guint64 duration,
                             gpointer user_data)
    {
      AppData *gApp = (AppData *) user_data;
      if (gApp->results.empty())
        return;

      std::vector<DetectedObject> detectedObjects = gApp->results;

      #ifdef PROFILING
      struct timeval preprocessStart, preprocessEnd;
      gettimeofday(&preprocessStart, NULL);
      #endif

      cairo_set_source_rgb(cr, 1, 0, 0);
      cairo_set_line_width(cr, 1.0);
  
      for (int i = 0; i < detectedObjects.size(); i++) {
        int x1 = detectedObjects.at(i).x;
        int y1 = detectedObjects.at(i).y;
        int w = detectedObjects.at(i).width;
        int h = detectedObjects.at(i).height;
        cairo_rectangle(cr, x1, y1, w, h);
        cairo_move_to(cr, x1 + 5, y1 + 10);
        cairo_show_text(cr, gApp->className.at(detectedObjects.at(i).classId).c_str());
      }
      cairo_stroke(cr);

      #ifdef PROFILING
      gettimeofday(&preprocessEnd, NULL);
      gApp->ppDuration.cairoIter += 1;
      gApp->ppDuration.cairoTotalDuration += 1000.0 * (preprocessEnd.tv_sec-preprocessStart.tv_sec) 
                                          + (preprocessEnd.tv_usec-preprocessStart.tv_usec) / 1000.0;
      #endif
    }
};


int cmdParser(int argc, char **argv, std::string &model, std::string &image)
{
  int c;
  int optionIndex;
  static struct option longOptions[] = {
    {"help",          no_argument,       0, 'h'},
    {"model_path",    required_argument, 0, 'm'},
    {"image_path",    required_argument, 0, 'i'},
    {0,               0,                 0,   0}
  };

  while ((c = getopt_long(argc,
                          argv,
                          "hm:i:",
                          longOptions,
                          &optionIndex)) != -1) {
    switch (c)
    {
      case 'h':
        std::cout << "Help Options:" << std::endl
                << std::setw(25) << std::left << "  -h, --help"
                  << std::setw(25) << std::left << "Show help options"
                  << std::endl << std::endl
                  << "Application Options:" << std::endl

                  << std::setw(25) << std::left << "  -m, --model_path"
                  << std::setw(25) << std::left
                  << "Select path to the model" << std::endl

                  << std::setw(25) << std::left << "  -i, --image_path"
                  << std::setw(25) << std::left
                  << "Select image path" << std::endl;
        return 1;
 
      case 'm':
        model = optarg;
        break;
      
      case 'i':
        image = optarg;
        break;
    }
  }
  return 0;
}


int main(int argc, char **argv)
{
  std::string model;
  std::string image;
  if (cmdParser(argc, argv, model, image))
    return 0;

  if (model.size() == 0) {
    log_error("Provide a model path after -m option\n");
  }
  if (image.size() == 0) {
    log_error("Select an image path after -i option\n");
  }

  YOLOv8 example(model, image);
  example.run();

  return 0;
}