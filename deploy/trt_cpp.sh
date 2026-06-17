set -e  # 任何一步失败就停止


export CUDA_VISIBLE_DEVICES=1
CPP/build/hpenet_trt_infer --engine trt_model_fp32.engine --num_files 10 --stats CPP/stats.json --data_dir ../data/RadarClassi/radarfull/raw


./build/hpenet_onnx_infer --data_dir <DIR> --output_dir <DIR> --onnx <MODEL> [--stats_file stats.json]