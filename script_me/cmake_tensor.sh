# linux trt8.6
cd deploy/CPP && rm -rf build && mkdir build && cd build
cmake .. -G "Unix Makefiles" \
  -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
make -j$(nproc)
ctest --output-on-failure


# linux trt10.16
cd deploy/CPP && rm -rf build && mkdir build && cd build
cmake .. -G "Unix Makefiles" \
  -DTENSORRT_ROOT=/usr/local/TensorRT-10.16.0.34 \
  -DCMAKE_CUDA_ARCHITECTURES="120"
make -j$(nproc)
ctest --output-on-failure


# windows msvc2019 trt10.16
cd deploy\CPP
rmdir /s /q build
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 ^
  -DTENSORRT_ROOT=C:/TensorRT-10.16.1.11 ^
  -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build . --config Release -j
ctest --output-on-failure -C Release
