# BNN 训练与权重导出（Issue-0004）

当前阶段先建立可复现的导出与一致性校验链路。

## 1. 导出权重

```bash
./scripts/export_bnn_weights.py
```

输出：
- `src/bnn_weights.h`：C 推理使用
- `docs/bnn_weights.json`：一致性校验使用

## 2. 编译推理工具

```bash
make
```

输出包含：
- `score_cli`：单样本打分工具

## 3. Python vs C 一致性校验

```bash
./scripts/check_consistency.py --samples 200
```

通过条件：
- `mismatches=0`
- `result=PASS`

## 4. 后续替换策略

当训练管线准备好后，只需要让训练侧产出同格式 `src/bnn_weights.h`，无需改动推理主逻辑。
