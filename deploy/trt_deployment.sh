# 构建 FP32 engine
LD_LIBRARY_PATH=/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib:/usr/local/cuda-11.8/lib64 \
python deploy/trt_build.py

# 推理
LD_LIBRARY_PATH=... python deploy/trt_inference.py --engine deploy/trt_model_fp32.engine --compare
