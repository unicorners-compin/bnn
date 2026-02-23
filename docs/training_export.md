# BNN 训练与权重导出（Issue-0004）

当前阶段已建立可复现的导出、热加载与一致性校验链路。

## 1. 导出权重

```bash
./scripts/export_bnn_weights.py --model-id 0 --bias 0 --bin models/model_0.bnnw
```

输出：
- `src/bnn_weights.h`：C 推理使用
- `docs/bnn_weights.json`：一致性校验使用
- `models/model_0.bnnw`：运行时热加载使用（可加载多个模型）

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

## 4. 多模型热加载与路由示例

```bash
# 导出第二个模型
./scripts/export_bnn_weights.py --model-id 1 --seed 0x123456789ABCDEF --bias 7 --bin models/model_1.bnnw --header /tmp/ignore.h --json /tmp/ignore.json

# 运行时加载 model_1，并根据前64B中的配置位选模型（默认读取 config[0]）
./score_cli --load-model models/model_1.bnnw --show-model /tmp/input.bin
```

## 5. 后续替换策略

当训练管线准备好后，只需要让训练侧产出同格式 `src/bnn_weights.h`，无需改动推理主逻辑。
