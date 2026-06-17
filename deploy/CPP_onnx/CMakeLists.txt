cmake_minimum_required(VERSION 3.18)
project(hpenet_onnx_infer LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use local ORT installation
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/onnxruntime")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

# Use local Eigen3
set(EIGEN3_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")

# Executable
add_executable(hpenet_onnx_infer
    onnx_inference.cpp
    main.cpp
)

target_include_directories(hpenet_onnx_infer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../CPP_trt/include  # tinyply
)

# Link ONNX Runtime
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_link_libraries(hpenet_onnx_infer PRIVATE onnxruntime)

# RPATH
if(UNIX)
    set_target_properties(hpenet_onnx_infer PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
        INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    )
endif()
