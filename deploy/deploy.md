现在要部署训练好的HPENet点云语义分割模型：

1.数据路径是data/RadarClassi/radrafull,数据的预处理和加载过程参考examples/segmentation/main.py:L53-77,L642-689
2.使用log/radar/radar-train-hpenet-l-ngpus1-20260515-013127-HXWMALkaAC4GiUWjNV5c3g/路径下的pth文件和其它需要的文件
3.进行onnx部署的python源码编写
4.将部署源码保存在deploy文件夹下
5.注意是否需要自定义算子