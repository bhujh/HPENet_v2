# 查看当前 core 文件大小限制，如果返回 0 说明未开启
ulimit -c 
# 临时开启不限制大小的 core 文件生成
ulimit -c unlimited

# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -X faulthandler examples/segmentation/main.py --cfg cfgs/radar/hpenet-ll.yaml dataset.train.voxel_max=8000 dataset.train.variable=True dataset.val.voxel_max=8000   dataset.val.variable=True batch_size=1

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python -X faulthandler examples/segmentation/main.py --cfg cfgs/radar/hpenet-ll.yaml