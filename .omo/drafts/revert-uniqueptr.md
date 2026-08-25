# Draft: Revert UniquePtr Changes

## 发现
用户指出 `nvinfer1::UniquePtr` 不存在于 TensorRT 官方 C++ API。

验证结果：
- TRT 8.6.1.6 头文件：无任何 `UniquePtr` 定义
- TRT 10 官方迁移指南：`destroy()` → `delete ObjectName`（factory 仍返回裸指针）
- TRT 10 示例中用的是 `SampleUniquePtr`（samples/common/ 中的工具类），不是 `nvinfer1::UniquePtr`

## 需要回退的文件
- deploy/CPP/include/trt_engine.h — 恢复裸指针 Runtime/Engine 成员
- deploy/CPP/src/trt_engine.cpp — 恢复手动 delete + 移除 .release()
- deploy/CPP/include/trt_inference.h — 恢复裸指针 context_ 成员
- deploy/CPP/src/trt_inference.cpp — 恢复 .reset() → 直接赋值 + 恢复手动 delete

## 保留的修改
- deploy/CPP/CMakeLists.txt — CUDAToolkit + MSVC CRT + DLL（正确）
- deploy/CPP/tests/CMakeLists.txt — 变量迁移（正确）
- deploy/CPP/cmake/FindTensorRT.cmake — TRT 10 命名 + Windows 路径（正确）

## 附加修复（Oracle P0）
- CMakeLists.txt: CMAKE_RUNTIME_OUTPUT_DIRECTORY 后备值
- FindTensorRT.cmake: Windows 上 TensorRT_ROOT 被忽略
