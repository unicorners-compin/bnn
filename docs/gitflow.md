# Git Flow 执行规范（本仓库）

## 1. 生命周期

1. 先开 `Issue`
2. 基于 `Issue` 新建分支开发
3. 提交代码并发起 `PR`
4. 评审通过后合并到 `main`
5. 关闭当前 `Issue`
6. 再开下一个 `Issue`

## 2. 命名规范

- Issue: `ISSUE-XXXX-<short-title>`
- 分支: `issue/XXXX-<short-title>`
- PR: `PR-XXXX-<short-title>`

## 3. 提交规范

- 提交信息格式：`type(scope): message`
- 常用 `type`：`feat` `fix` `perf` `test` `docs` `chore`
- 每个 PR 必须附带：
  - 性能结果（至少 P50/P99）
  - 准确性结果（与参考实现对比）
  - 风险与回滚说明

## 4. 本项目要求

- 任何功能开发都必须从 Issue 开始，不允许直接在 `main` 改动
- BNN 推理优化类 PR 必须附带基准数据
- 未完成基准或准确性验证，不可合并
