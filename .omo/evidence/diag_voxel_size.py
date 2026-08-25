"""Task 1: 诊断部署 voxel_size 实际生效值"""
import yaml, os, sys, re, subprocess

workdir = '/home/wangpeng/CODE/HPENet_v2-main'
os.chdir(workdir)
sys.path.insert(0, workdir)

# 1. trt_inference.py 默认 cfgPath
with open('deploy/trt_inference.py') as f:
    trt_src = f.read()
m = re.search(r"default='(cfgs/radar/.+?\.yaml)'", trt_src)
default_cfg = m.group(1) if m else 'NOT FOUND'
print(f'[trt_inference.py] default cfgPath = {default_cfg}')

# 2. 加载 YAML 并打印 radius
cfg_path = os.path.join(workdir, default_cfg)
print(f'[YAML] Loading: {cfg_path}')
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
radius = cfg.get('model', {}).get('encoder_args', {}).get('radius', 'KEY NOT FOUND')
print(f'[YAML] model.encoder_args.radius = {radius}')

# 3. 对照 hpenet-s / hpenet-b
for var in ['cfgs/radar/hpenet-s.yaml', 'cfgs/radar/hpenet-b.yaml']:
    with open(var) as f:
        c = yaml.safe_load(f)
    r = c.get('model', {}).get('encoder_args', {}).get('radius', 'N/A')
    print(f'[对照] {var} -> radius = {r}')

# 4. C++ main.cpp voxel_size 默认值
result = subprocess.run(['grep', '-n', 'voxel_size|0\\.1f|0\\.3f', 'deploy/CPP_trt1/src/main.cpp'],
                       capture_output=True, text=True)
print()
print('[C++] deploy/CPP_trt1/src/main.cpp 关键行:')
print(result.stdout)

# 5. 诊断 pipeline.cpp test_start
result2 = subprocess.run(['grep', '-n', 'test_start|n_total \\*', 'deploy/CPP_trt1/src/pipeline.cpp'],
                        capture_output=True, text=True)
print('[C++] deploy/CPP_trt1/src/pipeline.cpp test_start 关键行:')
print(result2.stdout)

# 6. 检查 preprocess_test 的 voxel_size 调用链
with open('deploy/common.py') as f:
    common = f.read()
m2 = re.search(r'def preprocess_test.*?(?=\ndef |\nclass |\Z)', common, re.DOTALL)
if m2:
    sig = m2.group(0)[:300]
    print(f'\n[Python] deploy/common.py preprocess_test 函数签名片段:')
    print(sig)

print()
print('=== 诊断结论 ===')
print(f'Python 部署默认 voxel_size = {radius} (来自 {default_cfg})')
print(f'C++ 部署默认 voxel_size = 0.1f (来自 deploy/CPP_trt1/src/main.cpp:34)')
print(f'差倍: {float(radius)}/0.1 = {float(radius)/0.1:.0f}x')
print(f'对齐目标: C++ main.cpp 默认值 0.1f -> {radius}f')
