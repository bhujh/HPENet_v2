# 修复 onnx_export.py：simplify 保留动态 npoints 维度

## TL;DR

> **Quick Summary**: `simplify_onnx()` 传了 `overwrite_input_shapes`，导致 onnxsim 把动态 `npoints` 维度替换成静态值 3500，TRT build 报 `kMIN dimensions in profile 0 are [1,1024,3] but input has static dimensions [1,3500,3]`。
> **Deliverables**: 修改 `deploy/onnx_export.py` 的 `simplify_onnx()`，移除 `overwrite_input_shapes` 参数
> **Estimated Effort**: Quick
> **Parallel Execution**: NO - single task
> **Critical Path**: Task 1 → 验证

---

## Context

### 错误信息
```
[TRT] [E] Error Code 4: Internal Error (pos: kMIN dimensions in profile 0 are
[1,1024,3] but input has static dimensions [1,3500,3].)
```

### 根因
- `onnx_export.py` 的 `simplify_onnx()` 调用 `simplify(onnx_model, overwrite_input_shapes=input_shapes)`
- `overwrite_input_shapes` 语义 = 用具体 shape **覆盖**模型输入定义
- onnxsim 将动态 `npoints` 维度替换为静态 3500 → 简化后模型输入变静态
- `trt_build.py` 的 optimization profile 要求动态 `(1, [1024-6000], 3)` → 静态输入 vs 动态 profile 矛盾 → build 失败

### 已证实的正确做法
手动调用 `simplify(m)`（不传 overwrite_input_shapes）：
- ✅ 动态维度保留（pos: `[1, 'npoints', 3]`）
- ✅ Constant→initializer 折叠正常（51/51 InstanceNormalization）
- ✅ TRT FP16 build 成功（14.6 MB engine）

---

## Work Objectives

### Core Objective
修改 `simplify_onnx()`，移除 `overwrite_input_shapes`，保留动态维度，使简化后的 ONNX 可被 TRT 动态 profile 构建。

### Concrete Deliverables
- `deploy/onnx_export.py`: `simplify_onnx()` 不再传 `overwrite_input_shapes`

### Definition of Done
- [ ] `python -m py_compile deploy/onnx_export.py` 通过
- [ ] 重新导出 ONNX 后 pos 输入维度为 `[1, 'npoints', 3]`（动态）
- [ ] TRT build 成功

### Must NOT Have
- 不要改动 `trt_build.py` 的 profile（动态维度是正确设计）
- 不要改动 `onnx_backend.py`（已修复）
- 不要删除简化步骤（Constant→initializer 是 TRT 必需）

---

## TODOs

- [ ] 1. 修改 `deploy/onnx_export.py` 的 `simplify_onnx()`

  **What to do**:
  - 删除 `input_shapes` dict 定义（L145-148）
  - 删除 `actual_inputs`/`missing` 守卫代码（L150-157）
  - `simplify(onnx_model, overwrite_input_shapes=input_shapes)` 改为 `simplify(onnx_model)`
  - 更新函数 docstring：说明不要传 overwrite_input_shapes 的原因（会破坏动态维度）
  - 保留 `num_points`/`num_features` 参数签名（虽然不再用于 input_shapes，保持调用兼容）

  **Must NOT do**:
  - 不修改其他函数
  - 不改 simplify 的返回/错误处理逻辑

  **Recommended Agent Profile**:
  - **Category**: `quick`（单文件小改动）
  - **Skills**: 无需额外技能

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `deploy/onnx_export.py:128-172` - `simplify_onnx()` 完整函数，需要修改的范围
  - `deploy/onnx_export.py:290-307` - `main()` 中调用 `simplify_onnx()` 的方式（参数不变）

  **Acceptance Criteria**:
  - [ ] `python -m py_compile deploy/onnx_export.py` 通过
  - [ ] grep 确认 `overwrite_input_shapes` 不再出现在文件中

  **QA Scenarios**:
  ```
  Scenario: 重新导出并验证动态维度保留
    Tool: Bash
    Preconditions: checkpoint 存在（log/radar/radar-train-hpenet-ll-ngpus1-20260625-144233-c5U2epnpA9JLFW53JxxUSj/checkpoint/..._ckpt_best.pth）
    Steps:
      1. python deploy/onnx_export.py --output /tmp/fix_verify.onnx --simplified_output /tmp/fix_verify_sim.onnx
      2. python -c "import onnx; m=onnx.load('/tmp/fix_verify_sim.onnx'); print([d.dim_value if d.HasField('dim_value') else d.dim_param for d in m.graph.input[0].type.tensor_type.shape.dim])"
      3. 断言输出包含 'npoints'（动态）
    Expected Result: pos 输入维度为 [1, 'npoints', 3]
    Failure Indicators: pos 输入维度为 [1, 3500, 3]（静态）→ 未修复
    Evidence: /tmp/fix_verify_sim.onnx + 终端输出
  ```

  **Commit**: NO（等验证后一起提交）

---

## Final Verification Wave

- [ ] F1. TRT build 验证 — `quick`
  `python deploy/trt_build.py --onnx /tmp/fix_verify_sim.onnx --output /tmp/fix_verify.engine --fp16 --opt_n 3500 --max_n 6000`
  期望：Engine saved（无 kMIN/kMAX 错误、无 layernorm overflow 警告）
  输出：`Build [PASS/FAIL] | Warnings [layernorm: N, initializer: N] | VERDICT`

---

## Commit Strategy

- 用户确认后：`fix(deploy): 保留 ONNX 动态维度，移除 overwrite_input_shapes`
  文件: deploy/onnx_export.py
  （用户明确允许 git 操作后才能提交——AGENTS.md 禁止未经允许执行 git 操作）

---

## Success Criteria

### Verification Commands
```bash
python -m py_compile deploy/onnx_export.py
python deploy/onnx_export.py --output /tmp/fix_verify.onnx --simplified_output /tmp/fix_verify_sim.onnx
python deploy/trt_build.py --onnx /tmp/fix_verify_sim.onnx --output /tmp/fix_verify.engine --fp16 --opt_n 3500 --max_n 6000
```

### Final Checklist
- [ ] simplify 后 pos 输入保持动态 `[1, 'npoints', 3]`
- [ ] TRT FP16 build 成功
- [ ] 无 layernorm overflow 警告
