#!/bin/sh
import os
import sys
from ubuntu_utils import cmd_result, ensure_base_env, get_job

g_jobs = 4

if not os.path.exists('build'):
    os.system('mkdir build')

os.system('rm -rf build/CMakeCache.txt')

cmd = 'cd build && cmake ..'
cmd += ' -DMMDEPLOY_BUILD_SDK=ON '
cmd += ' -DMMDEPLOY_BUILD_EXAMPLES=ON '
cmd += ' -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON '
cmd += ' -DMMDEPLOY_TARGET_BACKENDS=trt '
cmd += ' -DTENSORRT_DIR="/home/zph/3rdParty/TensorRT-8.4.1.5" '
cmd += ' -Dpplcv_DIR="/home/zph/3rdParty/mmdeploy-dep/ppl.cv/cuda-build/install/lib/cmake/ppl" '
cmd += ' -Dpplnn_DIR="/home/zph/3rdParty/mmdeploy-dep/ppl.nn/pplnn-build/install/lib/cmake/ppl" '
cmd += ' -DMMDEPLOY_TARGET_DEVICES=cuda '
os.system(cmd)

# os.system('cd build && make -j {}'.format(g_jobs))
os.system('cd build && make -j {} && make install'.format(g_jobs))