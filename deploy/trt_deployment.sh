#/bin/bash

set -e  # 任何一步失败就停止

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 构建 FP32 engine
# python deploy/trt_build.py

# 推理
python deploy/trt_inference.py --engine deploy/trt_model_fp32.engine --compare
