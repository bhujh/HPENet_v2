# FindTensorRT.cmake
# 查找 TensorRT 库（nvinfer, nvinfer_plugin, nvonnxparser）
#
# 用法:
#   find_package(TensorRT REQUIRED)
#   或指定根目录: cmake .. -DTensorRT_ROOT=/path/to/TensorRT
#                (CMake 3.28+ 策略 CMP0144, 旧版用 TENSORRT_ROOT)
#
# 输出变量:
#   TensorRT_FOUND          - 是否找到
#   TensorRT_INCLUDE_DIRS   - 头文件路径
#   TensorRT_LIBRARIES      - 库文件列表
#   TensorRT_VERSION        - 版本字符串 (如 "8.6.1.6")
#   TensorRT_ROOT           - 根目录

# ---------------------------------------------------------------------------
# 搜索路径优先级:
#   1. 用户指定的 TensorRT_ROOT (CMake 3.28+) 或 TENSORRT_ROOT (旧版)
#   2. 环境变量 TENSORRT_ROOT
#   3. 常见安装路径
# ---------------------------------------------------------------------------

# 优先使用用户指定的根目录 (兼容 CMake 3.28+ 策略 CMP0144)
if(NOT TensorRT_ROOT)
  if(DEFINED TENSORRT_ROOT)
    set(TensorRT_ROOT "${TENSORRT_ROOT}")
  elseif(DEFINED ENV{TENSORRT_ROOT})
    set(TensorRT_ROOT "$ENV{TENSORRT_ROOT}")
  endif()
endif()

if(TensorRT_ROOT)
  # 显式指定根目录时，搜索标准子目录结构
  # TensorRT 典型结构:
  #   TensorRT-{version}/
  #   ├── include/NvInfer.h
  #   └── lib/libnvinfer.so
  set(_TRT_SEARCH_PATHS
    "${TensorRT_ROOT}"
    "${TensorRT_ROOT}/targets/x86_64-linux-gnu"
  )
else()
  # 常见安装路径
  set(_TRT_SEARCH_PATHS
    /usr/local/TensorRT-8.6.1.6
    /usr/local/TensorRT-8.5.3.1
    /usr/local/TensorRT-8.4.3.1
    /usr/local/TensorRT-10.9.0.34
    /usr/local/TensorRT-10.16.0.34
    /usr/local/TensorRT-10.15.1.34
    /usr/local/TensorRT
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/TensorRT
  )
endif()

if(WIN32)
  # Windows 常见安装路径
  list(APPEND _TRT_SEARCH_PATHS
    "C:/TensorRT-10.16.1.11"
    "C:/TensorRT-10.16.0.34"
    "C:/TensorRT-10.9.0.34"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT"
  )
endif()

# ---------------------------------------------------------------------------
# 查找头文件: NvInfer.h (TensorRT 8.x+/10.x 使用大写文件名)
# ---------------------------------------------------------------------------
find_path(TensorRT_INCLUDE_DIR NvInfer.h
  PATHS ${_TRT_SEARCH_PATHS}
  PATH_SUFFIXES include
  DOC "TensorRT 头文件路径"
)

# ---------------------------------------------------------------------------
# 查找库文件
# ---------------------------------------------------------------------------
find_library(TensorRT_LIBRARY_NVINFER nvinfer
  PATHS ${_TRT_SEARCH_PATHS}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
  DOC "TensorRT 核心库 (nvinfer)"
)

find_library(TensorRT_LIBRARY_PLUGIN nvinfer_plugin
  PATHS ${_TRT_SEARCH_PATHS}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
  DOC "TensorRT 插件库 (nvinfer_plugin)"
)

find_library(TensorRT_LIBRARY_PARSER nvonnxparser
  PATHS ${_TRT_SEARCH_PATHS}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
  DOC "TensorRT ONNX 解析器 (nvonnxparser)"
)

# ---------------------------------------------------------------------------
# 处理 Found / Not Found
# ---------------------------------------------------------------------------
mark_as_advanced(
  TensorRT_INCLUDE_DIR
  TensorRT_LIBRARY_NVINFER
  TensorRT_LIBRARY_PLUGIN
  TensorRT_LIBRARY_PARSER
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  REQUIRED_VARS
    TensorRT_INCLUDE_DIR
    TensorRT_LIBRARY_NVINFER
  FAIL_MESSAGE
    "TensorRT 未找到！请指定 -DTENSORRT_ROOT=/path/to/TensorRT (e.g., /usr/local/TensorRT-8.6.1.6 or C:/TensorRT-10.16.1.11)"
)

if(TensorRT_FOUND)
  # 组装输出变量
  set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")
  set(TensorRT_LIBRARIES
    "${TensorRT_LIBRARY_NVINFER}"
    "${TensorRT_LIBRARY_PLUGIN}"
    "${TensorRT_LIBRARY_PARSER}"
  )

  # 提取 TensorRT ROOT (仅当未指定时自动推导)
  if(NOT TensorRT_ROOT)
    get_filename_component(_TRT_LIB_DIR "${TensorRT_LIBRARY_NVINFER}" DIRECTORY)
    get_filename_component(_TRT_LIB_DIR "${_TRT_LIB_DIR}" DIRECTORY)
    set(TensorRT_ROOT "${_TRT_LIB_DIR}")
    unset(_TRT_LIB_DIR)
  endif()

  # 读取版本号
  if(EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _NVINFER_VER_STR
      REGEX "#define NV_TENSORRT_MAJOR|#define NV_TENSORRT_MINOR|#define NV_TENSORRT_PATCH|#define NV_TENSORRT_BUILD"
    )
    string(REGEX REPLACE ".*NV_TENSORRT_MAJOR[ ]+([0-9]+).*" "\\1" TRT_MAJOR "${_NVINFER_VER_STR}")
    string(REGEX REPLACE ".*NV_TENSORRT_MINOR[ ]+([0-9]+).*" "\\1" TRT_MINOR "${_NVINFER_VER_STR}")
    string(REGEX REPLACE ".*NV_TENSORRT_PATCH[ ]+([0-9]+).*" "\\1" TRT_PATCH "${_NVINFER_VER_STR}")
    string(REGEX REPLACE ".*NV_TENSORRT_BUILD[ ]+([0-9]+).*" "\\1" TRT_BUILD "${_NVINFER_VER_STR}")
    set(TensorRT_VERSION "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}.${TRT_BUILD}")
    unset(_NVINFER_VER_STR)
    unset(TRT_MAJOR)
    unset(TRT_MINOR)
    unset(TRT_PATCH)
    unset(TRT_BUILD)
  else()
    set(TensorRT_VERSION "unknown")
  endif()

  # 打印摘要
  message(STATUS "Found TensorRT: ${TensorRT_INCLUDE_DIRS}")
  message(STATUS "  Root:      ${TensorRT_ROOT}")
  message(STATUS "  Version:   ${TensorRT_VERSION}")
  message(STATUS "  Libraries: ${TensorRT_LIBRARIES}")

  # 标记高级变量
  mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARIES)
endif()

# 清理临时变量
unset(_TRT_SEARCH_PATHS)
