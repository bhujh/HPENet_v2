# 锁定 oh-my-openagent 插件版本

## TL;DR

> **Quick Summary**: 将 `opencode.json` 中插件引用从 `@next` 改为 `@4.5.12`，锁定版本。
> 
> **Deliverables**:
> - `~/.config/opencode/opencode.json` — 插件版本锁定
> 
> **Estimated Effort**: Quick
> **Parallel Execution**: NO — 单任务
> **Critical Path**: 无依赖，直接执行

---

## Context

### Original Request
安装 oh-my-openagent v4.5.12 后，`opencode.json` 中插件注册写的是 `"oh-my-openagent@next"`，需要锁定为 `"oh-my-openagent@4.5.12"`。

---

## Work Objectives

### Core Objective
将 `~/.config/opencode/opencode.json` 中第 103 行的 `"oh-my-openagent@next"` 改为 `"oh-my-openagent@4.5.12"`。

### Concrete Deliverables
- 修改 `opencode.json` 中 `plugin` 数组的一行

### Must Have
- 版本号精确锁定为 `4.5.12`
- 其余配置不受影响

### Must NOT Have
- 不修改 provider 配置
- 不修改其他任何文件

---

## TODOs

- [ ] 1. 锁定插件版本号

  **What to do**:
  - 修改 `~/.config/opencode/opencode.json`
  - 将 `"plugin": ["oh-my-openagent@next"]` 改为 `"plugin": ["oh-my-openagent@4.5.12"]`

  **Must NOT do**:
  - 不修改 provider 配置
  - 不修改 `oh-my-openagent.json`

  **Recommended Agent Profile**:
  - **Category**: `quick` — 单文件单行修改

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocked By**: None

  **References**:
  - `~/.config/opencode/opencode.json:103` — 当前插件注册行

  **Acceptance Criteria**:
  - [ ] `grep "oh-my-openagent@4.5.12" ~/.config/opencode/opencode.json` 返回匹配
  - [ ] `grep "oh-my-openagent@next" ~/.config/opencode/opencode.json` 无匹配

  **QA Scenarios**:
  ```
  Scenario: 版本号已锁定
    Tool: Bash
    Steps:
      1. grep '"oh-my-openagent@4.5.12"' ~/.config/opencode/opencode.json
      2. 确认返回包含该行
    Expected Result: 输出包含 "oh-my-openagent@4.5.12"
    Evidence: .omo/evidence/task-1-version-locked.txt

  Scenario: @next 已移除
    Tool: Bash
    Steps:
      1. grep '"oh-my-openagent@next"' ~/.config/opencode/opencode.json
    Expected Result: 无输出（exit code 1）
    Evidence: .omo/evidence/task-1-no-next.txt
  ```

  **Commit**: NO

---

## Commit Strategy
不提交 git — 此为 OpenCode 配置文件修改。

## Success Criteria
- `grep "4.5.12" ~/.config/opencode/opencode.json` 有输出
- `grep "@next" ~/.config/opencode/opencode.json` 无输出
