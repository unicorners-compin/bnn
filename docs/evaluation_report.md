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
