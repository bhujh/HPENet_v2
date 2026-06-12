# tests/CMakeLists.txt
# GoogleTest 测试子目录
#
# 每个测试文件独立注册为可执行文件，自动链接 gtest/gtest_main
# enable_testing() 已在顶层 CMakeLists.txt 调用

# 辅助函数: 注册一个 CUDA 测试
function(add_gpu_test TEST_NAME TEST_SOURCE)
  add_executable(${TEST_NAME} ${TEST_SOURCE} ${ARGN})
  target_include_directories(${TEST_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}   # needed when .cpp sources use CUDA Runtime API
  )
  target_link_libraries(${TEST_NAME} PRIVATE
    gtest gtest_main
    ${TensorRT_LIBRARIES} CUDA::cudart
  )
  set_target_properties(${TEST_NAME} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# ===========================================================================
# 在此处注册测试:
# add_gpu_test(test_something test_something.cpp)
# ===========================================================================

# ------------------------------------------------------------------
# test_fnv_hash — FNV-1 hash CUDA kernel vs Python golden data
# ------------------------------------------------------------------
add_gpu_test(test_fnv_hash test_fnv_hash.cu)
target_sources(test_fnv_hash PRIVATE ${CMAKE_SOURCE_DIR}/src/kernels/fnv_hash.cu)
target_compile_definitions(test_fnv_hash PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

# 将测试数据复制到构建目录 (与测试工作目录一致)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data/)

# scatter_mean GPU kernel 测试
add_gpu_test(test_scatter_mean test_scatter_mean.cu ../src/kernels/scatter_mean.cu)

# ------------------------------------------------------------------
# test_voxelize — voxelization (FNV hash → sort → unique → sub-cloud)
# ------------------------------------------------------------------
add_gpu_test(test_voxelize test_voxelize.cu)
target_sources(test_voxelize PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/voxelize.cu
    ${CMAKE_SOURCE_DIR}/src/kernels/fnv_hash.cu
    ${CMAKE_SOURCE_DIR}/src/voxelizer.cpp
    ${CMAKE_SOURCE_DIR}/src/random_util.cpp
)
target_compile_definitions(test_voxelize PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
