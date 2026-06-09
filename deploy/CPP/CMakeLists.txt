cmake_minimum_required(VERSION 3.18)
project(hpenet_trt_infer LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 策略: 抑制 CMake 3.28+ 警告
# CMP0146: FindCUDA 模块已移除 (使用 find_package(CUDA) 仍有效)
# CMP0144: find_package 使用大写 <PACKAGENAME>_ROOT
cmake_policy(SET CMP0146 OLD)
cmake_policy(SET CMP0144 NEW)

# GPU 架构: 默认 Ampere+ (sm80, sm86, sm89)
# 跨平台使用时可通过 -DCMAKE_CUDA_ARCHITECTURES=... 覆盖
# 例如 RTX 5090 (Blackwell) 使用 sm120, RTX 5060 使用 sm120
set(CMAKE_CUDA_ARCHITECTURES "80;86;89" CACHE STRING
    "CUDA architecture codes (e.g., 80;86;89 for Ampere/Ada, 120 for Blackwell)")

# 查找 CUDA
find_package(CUDA REQUIRED)

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
enable_testing()
add_subdirectory(tests)

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
    src/voxelizer.cpp
    src/random_util.cpp
    src/kernels/voxelize.cu
    src/kernels/fnv_hash.cu
    src/kernels/scatter_mean.cu
)
target_include_directories(hpenet_trt_infer PRIVATE include ${CUDA_INCLUDE_DIRS})
target_link_libraries(hpenet_trt_infer PRIVATE ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})

# RPATH 配置 (仅 Unix; Windows 使用 DLL 搜索路径)
if(UNIX)
  set_target_properties(hpenet_trt_infer PROPERTIES
    BUILD_RPATH "${TensorRT_ROOT}/lib:${CUDA_TOOLKIT_ROOT_DIR}/lib64"
    INSTALL_RPATH "${TensorRT_ROOT}/lib:${CUDA_TOOLKIT_ROOT_DIR}/lib64"
  )
endif()
