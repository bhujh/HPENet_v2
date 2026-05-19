# 导出 ONNX（已完成）
python deploy/onnx_export.py

# 推理验证
python deploy/onnx_inference.py --num_files 10
