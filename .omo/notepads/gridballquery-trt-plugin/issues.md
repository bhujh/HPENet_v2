# Issues — gridballquery-trt-plugin

Problems and gotchas encountered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---
## 2026-08-22 Task5
- [记录不需处理] fp16 整网 logits maxdiff 0.61–6.12 超 3e-2 判据、5/58 帧一致率 99.38–99.5%<99.5%；triage 证实为 TRT FP16 tactic 选择差异（非 GridBallQuery 语义），mIoU 差 0.0417pp ≤0.5pp 主仲裁通过。详见 deploy/evidence/gridballquery-e2e.md。
- [待用户决定] 构建产物 5 个新 engine（gridbq_fp32/fp16、inc_fp32/fp16、inc_fp16_b）与 timing.cache 更新未提交。
