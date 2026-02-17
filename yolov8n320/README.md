# YOLOv8 N 320 demo for i.MX 95
This section provides an application which takes a static jpeg image as source, and draw rectangles around objects from the input image.<br>
The application is complete vision pipeline with image post-processing, YOLOv8 N 320 inference running on Neutron, and a post-processing of the output tensors.<br>
3 files are provided in this folder:
- yolov8n_320_tfilte.cpp code which needs to be compiled for i.MX 95 board
- CMakeLists.txt to compile the source code
- A README to explain the steps which needs to be followed to execute this demo

# How to run the example
## Download YOLOv8 N 320 model
Install Ultralytics tool following installation instruction from git repository https://github.com/ultralytics/ultralytics/tree/main.<br>
The version tested is `8.3.87`, model architecture might differ with later versions which could be incompatible with eIQ ToolKit version (1.15.0) used to convert the model for i.MX 95.<br>
To generate the YOLOv8 N 320 variant, execute the following command:
```bash
yolo export model=yolov8n.pt imgsz=320 format=tflite int8
```
It will generate a folder with different versions of the model. The model used for this demo needs to be the full integrer quantized version to be compatible with neutron hardware: `yolov8n_saved_model/yolov8n_full_integer_quant.tflite`
## Convert YOLOv8 N 320 model to run on Neutron
### eIQ Toolkit install
Download eIQ Toolkit install package from DOWNLOADS section of NXP eIQ website. Type `1.15.0` on the Downloads search bar and select the eIQ Toolkit 1.15.0 Installer for Ubuntu. No additional Extension is needed.<br>
Login to an NXP account is required to start the download, it will be possible to create one directly when the installer is selected.<br>
Then open a terminal and enter following commands:
```bash
# paths below to be adapted according to package version
# make package executable and execute with root privilege
chmod a+x ~/Downloads/eiq-toolkit-v1.15.0.<...>.deb.bin
sudo ~/Downloads/eiq-toolkit-v1.15.0.<...>.deb.bin
```
### eIQ Toolkit environment setup
eIQ Toolkit environment configures specific python interpreter and packages to be used. It shall be done only once within same shell session:
```bash
# adapt base path according to your install
EIQ_BASE="/opt/nxp/eIQ_Toolkit_v1.15.0"
EIQ_ENV="${EIQ_BASE}/bin/eiqenv.sh"
source "${EIQ_ENV}"
```
### Neutron converter
The model needs to be converted using `neutron-converter` to run on i.MX 95 NPU. Execute the following commands to convert the model:
```bash
cd $EIQ_BASE/neutron-tuning
./neutron-converter --input /path-to-model/yolov8n_full_integer_quant.tflite --target imx95 --use-python-prototype --output /path-to-save-converted-model/yolov8n_full_integer_quant-320-neutron.tflite
```

## Execution on target

Once the model has been converted with Neutron converter, the source code of the demo needs to be cross-compiled. Cross-compilation instructions can be found in the main page [documentation](../README.md). The instructions need to be followed in yolov8n320 folder.<br>
The generated binary, the converted model, and the image needs to be sent to the board.<br>
The demo can be launched on target with the following command:
```bash
yolov8n_320_tflite -m /root/yolov8n_full_integer_quant-320-neutron.tflite -i /path-to-jpeg-image
```
If the demo is run with a BSP release version above `scarthgap`, neutron zero-copy feature needs to be disabled for the demo to run properly:
```bash
export NEUTRON_ENABLE_ZERO_COPY=0
```