# Core ML feature support

MMDeploy support convert Pytorch model to Core ML and inference.

## Installation

To convert the model in mmdet, you need to compile libtorch to support custom operators such as nms (only needed in conversion stage). For MacOS 12 users, please install Pytorch 1.8.0, for MacOS 13 users, please install Pytorch 2.0.0+.

```bash
cd ${PYTORCH_DIR}
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=`which python` \
    -DCMAKE_INSTALL_PREFIX=install \
    -DDISABLE_SVE=ON
make install
```

## Usage

```bash
python tools/deploy.py \
    configs/mmdet/detection/detection_coreml_static-800x1344.py \
    /mmdetection_dir/configs/retinanet/retinanet_r18_fpn_1x_coco.py \
    /checkpoint/retinanet_r18_fpn_1x_coco_20220407_171055-614fd399.pth \
    /mmdetection_dir/demo/demo.jpg \
    --work-dir work_dir/retinanet \
    --device cpu \
    --dump-info
```
