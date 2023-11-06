# mmdeploy安装编译过程
```sh
cd mmdeploy

# 1. 安装依赖并编译相关库，以及示例程序
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc) #编译onnx
python3 tools/scripts/build_tensorrt_cmake.py #编译tensorrt

# 2.添加路径
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH # 加载onnxruntime的库
export LD_LIBRARY_PATH=/home/zph/3rdParty/TensorRT-8.4.1.5/lib:$LD_LIBRARY_PATH # 加载TensorRT的库

# 3. 对应模型生成推理引擎
# 3.1 对应模型生成onnxruntime的model
python tools/deploy.py \                     
    configs/mmseg/segmentation_tensorrt_static-512x512.py \
    deeplabv3plus_r18-d8_4xb2-80k_cityscapes-769x769.py \
    deeplabv3plus_r18-d8_769x769_80k_cityscapes_20201226_083346-f326e06a.pth \
    demo/resources/cityscapes.png \
    --work-dir mmdeploy_models/mmseg/trt \
    --device cuda \
    --show \
    --dump-info

# 3.2 对应模型生成tensorrt的model
python tools/deploy.py \                     
    configs/mmseg/segmentation_tensorrt_static-512x512.py \
    deeplabv3plus_r18-d8_4xb2-80k_cityscapes-769x769.py \
    deeplabv3plus_r18-d8_769x769_80k_cityscapes_20201226_083346-f326e06a.pth \
    demo/resources/cityscapes.png \
    --work-dir mmdeploy_models/mmseg/trt \
    --device cuda \
    --show \
    --dump-info

```
