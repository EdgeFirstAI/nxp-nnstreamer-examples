# YOLOv8n 640x640 Object Detection — EdgeFirst Optimized

Real-time YOLOv8n object detection on NXP i.MX processors using NNStreamer
pipelines with EdgeFirst optimizations for DMA-BUF zero-copy, CameraAdaptor
preprocessing offload, and EdgeFirst HAL quantized NMS.

## Prerequisites

Build the Yocto SDK and target image following the **EdgeFirst for i.MX User Manual** (Yocto Integration section). The SDK includes all required dependencies.

## Targets

| Binary | Platform | Description |
|--------|----------|-------------|
| `yolov8n_reference` | i.MX 95 / 8M Plus | Standard NXP pipeline baseline (manual NMS) |
| `yolov8n_ara2_reference` | Kinara Ara-2 PCIe | Standard NXP pipeline baseline (manual NMS) |
| `yolov8n_imx8mp` | i.MX 8M Plus | EdgeFirst cameraadaptor in VX-Delegate + DMA-BUF + HAL NMS |
| `yolov8n_ara2` | Kinara Ara-2 | edgefirstcameraadaptor (G2D + NEON) + HAL NMS |
| `yolov8n_imx95` | i.MX 95 | edgefirstcameraadaptor (G2D + NEON) + HAL NMS |

## Building

### i.MX 8M Plus

```bash
source <yocto-sdk-imx8mp>/environment-setup-armv8a-poky-linux
mkdir -p build-imx8mp && cd build-imx8mp
cmake ..
make yolov8n_reference yolov8n_imx8mp yolov8n_ara2 yolov8n_ara2_reference
```

### i.MX 95

```bash
source <yocto-sdk-imx95>/environment-setup-armv8a-poky-linux
mkdir -p build-imx95 && cd build-imx95
cmake ..
make yolov8n_reference yolov8n_imx95
```

## Deploy

```bash
# i.MX 8M Plus
scp build-imx8mp/yolov8n_reference build-imx8mp/yolov8n_imx8mp root@<board>:/usr/bin/
scp build-imx8mp/yolov8n_ara2 build-imx8mp/yolov8n_ara2_reference root@<board>:/usr/bin/

# i.MX 95
scp build-imx95/yolov8n_reference build-imx95/yolov8n_imx95 root@<board>:/usr/bin/
```

## Running

### i.MX 8M Plus

```bash
# Reference baseline
yolov8n_reference -m yolov8n_640x640.tflite -c /dev/video3 --platform imx8mp

# EdgeFirst optimized (instrumented, 1000 frames)
yolov8n_imx8mp -m yolov8n_640x640.tflite -c /dev/video3 -I -n 1000

# EdgeFirst headless (no display, edge IoT)
yolov8n_imx8mp -m yolov8n_640x640.tflite -c /dev/video3 -H -n 1000
```

### Ara-2

```bash
# Reference baseline
yolov8n_ara2_reference -m yolov8n.dvm -v video.mp4

# EdgeFirst (edgefirstcameraadaptor)
yolov8n_ara2 -m yolov8n.dvm -v video.mp4 -c edgefirst.yaml
```

### i.MX 95

```bash
# Reference baseline
yolov8n_reference -m yolov8n_640x640_converted.tflite --platform imx95

# EdgeFirst (edgefirstcameraadaptor, instrumented)
yolov8n_imx95 -m yolov8n_640x640_converted.tflite -I
```

> **Note:** On i.MX 95, set `NEUTRON_ENABLE_ZERO_COPY=0` if you observe incorrect detections.

## Command-Line Options

All binaries share a common set of flags:

```
-m, --model <path>        Model file (required)
-c, --camera <device>     Camera device (default: auto)
-v, --video <file>        Video file input
-i, --image <file>        Static image input (reference only)
-I, --instrumented        Detailed timing breakdown
-H, --headless            No display output
-n, --num-frames <N>      Stop after N frames (0=infinite)
--platform imx95|imx8mp   Platform selection (reference only)
```

## Documentation

See [EDGEFIRST.md](EDGEFIRST.md) for:
- EdgeFirst optimization details (DMA-BUF zero-copy, CameraAdaptor, HAL)
- Performance comparison (reference vs EdgeFirst)
- Pipeline architecture diagrams
- Troubleshooting
