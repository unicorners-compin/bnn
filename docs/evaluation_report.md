# BNN 时延与准确性评估报告（Issue-0005）

日期：2026-02-23
分支：`issue/0005-latency-accuracy-evaluation-report`

## 1. 测试命令

```bash
make clean && make
./eval
./bench
./scripts/check_consistency.py --samples 500
```

## 2. 推理一致性结果

`./eval` 输出：

- `default_backend=avx512`
- `scalar_score=66.0`
- `avx2_score=66.0`
- `avx512_score=66.0`
- `scalar_avx2_match=yes`
- `scalar_avx512_match=yes`

结论：三条路径输出一致。

## 3. Python vs C 一致性结果

`./scripts/check_consistency.py --samples 500` 输出：

- `samples=500`
- `mismatches=0`
- `result=PASS`

结论：导出权重下，Python 参考实现与 C 实现一致。

## 4. 时延结果

`./bench` 输出（500000 次）：

- scalar
  - `avg_latency_us=0.1266`
  - `p50_ns=106`
  - `p99_ns=113`
- avx2
  - `avg_latency_us=0.0991`
  - `p50_ns=78`
  - `p99_ns=85`
- avx512
  - `avg_latency_us=0.1047`
  - `p50_ns=83`
  - `p99_ns=87`

结论：当前实测远低于 `10us` 目标。

## 5. 说明

- 本轮准确性评估基于当前导出权重的函数一致性（Python vs C），尚未接入真实业务标签数据集的 AUC/F1。
- `bench` 使用 `clock_gettime` 做单次采样，结果可用于工程对比；后续可补充 `rdtsc`/绑核策略进一步降低测量噪声。

## 6. 验收状态

- 时延目标：通过（`<=10us`）
- 实现一致性：通过
- 离线任务精度（AUC/F1）：待接入真实数据后补充

## 7. 模型路由与切换时延（Issue-0007）

测试日期：2026-02-23  
说明：新增“前64B配置位路由模型、后1024B做BNN”的实测分布。

### 7.1 固定模型（不切换）

- 场景：`single_model_fixed`
- 后端：`avx2`
- 次数：`500000`
- `avg_ns=95.50`
- `p50_ns=75`
- `p90_ns=82`
- `p99_ns=83`
- `p999_ns=89`

### 7.2 按配置位每次切换模型（model0/model1交替）

- 场景：`route_switch_every_call`
- 后端：`avx2`
- 次数：`500000`
- `avg_ns=98.60`
- `p50_ns=79`
- `p90_ns=83`
- `p99_ns=87`
- `p999_ns=97`

结论：按配置位路由切模型开销很小，仍保持约 `80ns~100ns` 级。

### 7.3 文件热加载切换（每次重新加载模型文件）

- 场景：`hotload_switch_files`
- 次数：`20000`
- `avg_ns=2445.14`（`2.4451us`）
- `p50_ns=2412`
- `p90_ns=2474`
- `p99_ns=2503`
- `p999_ns=4528`

结论：文件热加载属于微秒级，不应放在在线请求路径。推荐启动阶段一次性加载全部模型到内存，在线仅做配置位路由。
