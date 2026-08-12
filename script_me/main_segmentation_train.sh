# 查看当前 core 文件大小限制，如果返回 0 说明未开启
ulimit -c 
# 临时开启不限制大小的 core 文件生成
ulimit -c unlimited

CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 python -X faulthandler examples/segmentation/main.py --cfg cfgs/radar/hpenet-ll.yaml