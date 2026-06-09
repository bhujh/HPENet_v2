set -e  # 任何一步失败就停止

CPP/build/hpenet_trt_infer --engine trt_model_fp32.engine --num_files 10 --stats CPP/stats.json --data_dir ../data/RadarClassi/radarfull/raw
