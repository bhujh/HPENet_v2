# Learnings - deploy-cpp-port

## Initial Setup
- Plan: .omo/plans/deploy-cpp-port.md
- 18 implementation tasks + 4 final verification tasks
- 4 waves: Wave 1 (T1-T7 parallel), Wave 2 (T8-T12 parallel), Wave 3 (T13-T15 serial), Wave 4 (T16-T18 parallel)

## Key Constraints
- No modification to deploy/*.py files
- No heavy dependencies (OpenCV, PCL, Eigen, Boost)
- No ONNX export or TRT engine building - only inference
- No Python bindings (no pybind11)
- C++17, CUDA 11.3+, sm80/sm86/sm89
- TensorRT 8.6.1.6 C++ API (nvinfer1::IRuntime, IExecutionContext, enqueueV3)
