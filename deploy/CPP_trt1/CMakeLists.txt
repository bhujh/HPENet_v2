cmake_minimum_required(VERSION 3.18)
project(hpenet_trt_infer LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC CRT: 匹配 TensorRT 预编译库的动态 CRT (/MD)
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# 策略: 抑制 CMake 3.28+ 警告
# CMP0144: find_package 使用大写 <PACKAGENAME>_ROOT
cmake_policy(SET CMP0144 NEW)

# GPU 架构: 默认 Ampere+ (sm80, sm86, sm89)
# 跨平台使用时可通过 -DCMAKE_CUDA_ARCHITECTURES=... 覆盖
# 例如 RTX 5090 (Blackwell) 使用 sm120, RTX 5060 使用 sm120
set(CMAKE_CUDA_ARCHITECTURES "80;86;89" CACHE STRING
    "CUDA architecture codes (e.g., 80;86;89 for Ampere/Ada, 120 for Blackwell)")

# 查找 CUDA (使用现代 CUDAToolkit 模块替代废弃的 FindCUDA)
find_package(CUDAToolkit REQUIRED)

# 查找 TensorRT
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)
message(STATUS "TensorRT version: ${TensorRT_VERSION}")

# GoogleTest (FetchContent)
include(FetchContent)
# FetchContent_Declare(
#   googletest
#   GIT_REPOSITORY https://github.com/google/googletest.git
#   GIT_TAG release-1.12.1
# )
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://gitee.com/mirrors/googletest.git
  GIT_TAG release-1.12.1
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(googletest)
#enable_testing()
#add_subdirectory(tests)

# 可执行文件
add_executable(hpenet_trt_infer
    src/main.cpp
    src/pipeline.cpp
    src/ply_reader.cpp
    src/trt_inference.cpp
    src/stats_reader.cpp
    src/preprocessor.cpp
    src/subcloud_utils.cpp
    src/trt_engine.cpp
    src/random_util.cpp
    src/logger.cpp
    src/trt_inference_wrapper.cpp
    src/cuda_utils.cu
    src/voxelizer.cu
    src/fnv_hash.cu
    src/scatter_mean.cu
)
target_include_directories(hpenet_trt_infer PRIVATE include ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(hpenet_trt_infer PRIVATE ${TensorRT_LIBRARIES} CUDA::cudart)

# RPATH 配置 (仅 Unix; Windows 使用 DLL 搜索路径)
if(UNIX)
  set_target_properties(hpenet_trt_infer PROPERTIES
    BUILD_RPATH "${TensorRT_ROOT}/lib:${CUDA_TOOLKIT_ROOT_DIR}/lib64"
    INSTALL_RPATH "${TensorRT_ROOT}/lib:${CUDA_TOOLKIT_ROOT_DIR}/lib64"
  )
elseif(WIN32)
  # 确保 RUNTIME_OUTPUT_DIRECTORY 有效（VS 多配置生成器下可能为空）
  if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  endif()
  # 复制 TRT DLL 到输出目录（TRT 10: nvinfer_10.dll 等）
  file(GLOB TRT_DLLS "${TensorRT_ROOT}/lib/*.dll")
  foreach(DLL ${TRT_DLLS})
    configure_file("${DLL}" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/" COPY_ONLY)
  endforeach()
  message(STATUS "Copied TensorRT DLLs to output directory")
endif()
