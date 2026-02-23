# 模型路由与热加载说明（Issue-0007）

## 1. 业务逻辑

- 输入固定 `1088B`
- 前 `64B` 为配置域：用于选择模型
- 后 `1024B` 为 payload：仅这部分进入 BNN `XNOR+POPCNT` 计算

## 2. 路由规则

当前默认规则：

- 读取 `config[0:8]` 的 64 位值
- 取 `(value & 0xFF) >> 0` 作为 `model_id`

可在运行时修改：

- `bnn_set_route_rule(mask, shift)`

例如：
- `mask=0xF0, shift=4` 表示使用第一个字节高4位作为模型编号。

## 3. 权重文件格式（`.bnnw`）

二进制结构（小端）：

1. `uint32 magic`，固定 `0x4D4E4E42`（BNNM）
2. `uint32 version`，固定 `1`
3. `uint32 model_id`
4. `uint32 words`，固定 `128`（对应 1024B payload）
5. `int32 bias`
6. `uint32 reserved`
7. `uint64 weights[128]`

## 4. C API

- `int bnn_load_model_file(const char *path);`
- `int bnn_set_default_model(uint32_t model_id);`
- `int bnn_set_route_rule(uint64_t mask, uint8_t shift);`
- `uint32_t bnn_pick_model_id(const uint8_t config[64]);`
- `uint32_t bnn_active_model_id(void);`

## 5. CLI 示例

```bash
# 导出模型0/1
./scripts/export_bnn_weights.py --model-id 0 --bin models/model_0.bnnw
./scripts/export_bnn_weights.py --model-id 1 --seed 0x123456789ABCDEF --bias 7 --bin models/model_1.bnnw --header /tmp/ignore.h --json /tmp/ignore.json

# 加载 model_1 后，按 config 位自动路由
./score_cli --load-model models/model_1.bnnw --show-model /tmp/in_model_0.bin
./score_cli --load-model models/model_1.bnnw --show-model /tmp/in_model_1.bin
```

示例输出：

- `model_id=0 score=-44.0`
- `model_id=1 score=-5.0`

说明：同 payload，不同配置位会路由到不同模型，得分不同。
