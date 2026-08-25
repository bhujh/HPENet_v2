# trt-version-compat: Learnings

## 2026-06-09 Session Start
- Plan: deploy/CPP/ cross-platform cmake for TRT 8.6/10.x on Linux & Windows
- All C++ source is already source-compatible with both TRT 8.6 and TRT 10.x
- Only 3 files need modification: FindTensorRT.cmake, CMakeLists.txt, trt_engine.cpp
- TRT 10.x Dims::d[] changed int32_t→int64_t but source-compatible (implicit conversion)
- RPATH is Linux-only, needs if(UNIX) guard for Windows
