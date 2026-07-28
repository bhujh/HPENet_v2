# linux trt8.6
cd deploy/CPP_trt && rm -rf build && mkdir build && cd build
cmake .. -G "Unix Makefiles" \
  -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
make -j$(nproc)
ctest --output-on-failure
# 执行：
cd ~/CODE/HPENet_v2-main
./deploy/CPP_trt/build/hpenet_trt_infer


# C接口的tensorrt部署代码
cd deploy/CPP_trt1 && rm -rf build && mkdir build && cd build
cmake .. -G "Unix Makefiles" -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
make -j$(nproc)
ctest --output-on-failure
# 执行：
cd ~/CODE/HPENet_v2-main
./deploy/CPP_trt1/build/hpenet_trt_infer "/home/wangpeng/CODE/HPENet_v2-main/deploy/trt_model_fp32.engine" "/home/wangpeng/CODE/HPENet_v2-main/deploy/CPP_trt/stats.json" "/home/wangpeng/CODE/HPENet_v2-main/data/RadarClassi/radarfull/raw/0000071.ply" "/home/wangpeng/CODE/HPENet_v2-main/deploy/CPP_trt1/output/baocun0000071.ply"


# 交叉编译trt/C接口代码
rm -rf build && mkdir build && cd build
cmake .. -G "Unix Makefiles" -DCMAKE_CUDA_FLAGS="-cudart shared" -DCMAKE_TOOLCHAIN_FILE=./toolchain.cmake -DTENSORRT_ROOT=/usr/local/TensorRT-aarch64-8.2.5.1 -DCMAKE_CUDA_ARCHITECTURES="87"
make -j$(nproc)
####################
cmake -B build-aarch64 -DCMAKE_TOOLCHAIN_FILE=aarch64-toolchain.cmake (-DTensorRT_ROOT=/usr/local/TensorRT-aarch64-8.5.3.1)
cmake --build build-aarch64 -j$(nproc)
./build-aarch64/hpenet_trt_infer
scp build-aarch64/hpenet_trt_infer adas@192.168.137.40:/home/adas/CODE/CPP_trt1/build-aarch64/
#################################


# windows msvc2019 trt10.16
cd deploy\CPP
rmdir /s /q build
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 \
  -DTENSORRT_ROOT=C:/TensorRT-10.16.1.11 \
  -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build . --config Release -j
ctest --output-on-failure -C Release



# CPP_onnx
cd deploy/CPP_onnx
rm -rf build && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

cd deploy/CPP_onnx
LD_LIBRARY_PATH=./lib ./build/hpenet_onnx_infer \
  --onnx ../../deploy/onnx_model.onnx \
  --data_dir ../../data/RadarClassi/radarfull/raw \
  --output_dir ./output \
  --stats_file stats.json
