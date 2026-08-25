# trt-version-compat: Decisions

## Adopted
1. Version detection: static_assert(NV_TENSORRT_MAJOR >= 8) + comments only (no #ifdef)
2. CMAKE_CUDA_ARCHITECTURES: cache variable defaulting to "80;86;89"
3. RPATH: wrapped in if(UNIX) for Windows safety
4. Windows TRT search paths: if(WIN32) block with C:/TensorRT-10.16.1.11 etc.
5. Keep find_package(CUDA) + CMP0146 OLD — no CMake modernization
6. Single commit for all 3 files
