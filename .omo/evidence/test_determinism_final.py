"""Task 2 最终验证: 每文件独立 seed 模型"""
import numpy as np, sys, os, hashlib
sys.path.insert(0, '/home/wangpeng/CODE/HPENet_v2-main')
os.chdir('/home/wangpeng/CODE/HPENet_v2-main')

N = 100
coord = np.random.RandomState(42).randn(N, 3).astype(np.float32) * 5.0
feat = np.random.RandomState(42).randn(N, 2).astype(np.float32)

from deploy.common import preprocess_test

# 测试 per-file seed 模型（新行为）
r1 = preprocess_test(coord.copy(), feat.copy(), voxel_size=0.3, seed=100)
r2 = preprocess_test(coord.copy(), feat.copy(), voxel_size=0.3, seed=100)

same = all(np.array_equal(a, b) for a, b in zip(r1[2], r2[2]))
h1 = hashlib.sha256(str([x.tolist() for x in r1[2]]).encode()).hexdigest()[:12]
h2 = hashlib.sha256(str([x.tolist() for x in r2[2]]).encode()).hexdigest()[:12]
print(f"sha256: {h1} vs {h2}")
print(f"每文件 seed=100 确定性: {same}")
assert same, "FAIL"

# 测试不同 seed (seed=100 vs seed=200) 应不同
r3 = preprocess_test(coord.copy(), feat.copy(), voxel_size=0.3, seed=200)
diff_seed = not all(np.array_equal(a, b) for a, b in zip(r1[2], r3[2]))
print(f"不同 seed 产生不同结果: {diff_seed}")
assert diff_seed, "不同 seed 应产生不同结果"

print("ALL PASS: per-file seed model works correctly")
with open('.omo/evidence/task-2-determinism.txt', 'w') as f:
    f.write(f"sha256_run1: {h1}\nsha256_run2: {h2}\ndeterministic: {same}\n")
    f.write("结论: preprocess_test(seed=100) 实现每文件独立确定性\n")
    f.write("修改: common.py +seed 参数, trt_inference.py/onnx_inference.py 传入 seed=100\n")
