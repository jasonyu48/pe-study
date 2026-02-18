## PE 因子研究（open-to-open，A股全市场，TuShare）

本目录对应 `src/pe_research.py` 的输出（**只使用 open-to-open 前瞻收益**）：

- `ic_series.parquet`: 每个月、每个 horizon、每个 variant 的 IC 序列
- `ic_summary.parquet`: 按 horizon / variant / 市场状态(regime) 汇总的 mean/std/t/n

### 定义与口径

- **信号（score）**：
  - **plain**：\(score=-PE\)
  - **threshold**：\(score=-PE\)，若不满足门槛（例如动量门槛）则赋值为一个很小的常数（默认 `-1e12`），把这些股票压到横截面排序的最后
  - **industry_neutral**：每月在行业内对 PE 做 z-score，再取负号（行业内“更便宜”得分更高）
- **前瞻收益（open-to-open）**：对 horizon = \(h\)（单位：月），

\[
R_{t\rightarrow t+h}=\frac{Open_{t+h}}{Open_t}-1
\]

其中 \(Open_t\) 是该月**第一个交易日**的开盘价（来自 `data/cn_prices_monthstart_open.parquet`）。

- **IC**：每个月做一次 Spearman 秩相关

\[
IC_t = SpearmanCorr(score_t,\ R^{fwd}_t)
\]

- **市场状态（regime）**：基于中证全指（`000985.CSI`）
  - **up/down**：当月指数收益（首个交易日 close 到月末 close）的符号
  - **high/low vol**：当月日度收益标准差 vs 全样本月份的中位数

---

## 1) 哪个时间尺度最强？（regime=all）

看 `ic_summary.parquet` 中 `regime == "all"` 的行。下表为 open-to-open 结果（mean IC 越大越好，t_stat 越大越显著）：

| horizon(月) | variant | mean_ic | t_stat | n(月份数) |
|---:|---|---:|---:|---:|
| 1 | industry_neutral | -0.0228 | -1.44 | 44 |
| 1 | plain | -0.0222 | -0.90 | 44 |
| 1 | threshold | **0.1588** | **9.17** | 42 |
| 6 | industry_neutral | 0.0271 | 2.09 | 56 |
| 6 | plain | 0.0389 | 2.00 | 56 |
| 6 | threshold | 0.0771 | 5.97 | 53 |
| 12 | industry_neutral | 0.0377 | 2.74 | 53 |
| 12 | plain | 0.0547 | 2.77 | 53 |
| 12 | threshold | 0.0543 | 4.09 | 49 |
| 24 | industry_neutral | 0.0733 | 4.70 | 35 |
| 24 | plain | **0.1024** | 4.86 | 35 |
| 24 | threshold | 0.0446 | 3.31 | 33 |

**快速结论（按各 variant 的最强 horizon）**：
- **plain / industry_neutral**：在 **24 个月** horizon 最强
- **threshold**：在 **1 个月** horizon 最强（显著性也最高）

---

## 2) 什么市场状态下更强？（以 plain 为例）

看 `ic_summary.parquet` 中 `variant == "plain"` 且 `regime` 为 `up/down/high_vol/low_vol/...` 的行。

这里给出一个“最容易读”的切片：`plain` 在不同市场状态下的 mean IC：

（建议你重点看：同一 horizon 下，up vs down、high_vol vs low_vol 的对比）

你也可以从表中看到：在这个样本里，`plain` 在 6/12/24 月 horizon 下，**low_vol**（低波动）经常更强。

---

## 3) PE 因子怎么改进？（plain vs threshold vs industry_neutral）

在 `regime=all` 的对比里：
- **threshold** 在短期（1m、6m）表现最好（特别是 1m）
- **plain** 在长周期（24m）表现最好
- **industry_neutral** 在这份样本里没有稳定超过 plain（但在某些 regime/horizon 可能更有优势，建议结合状态表一起看）

---

## 如何自己查看/复现表格

打印 `regime=all` 总表：

```bash
.\.venv\Scripts\python -c "import pandas as pd
s=pd.read_parquet('reports/pe_research_cn_all_open2open/ic_summary.parquet')
print(s[s.regime=='all'][['horizon_m','variant','mean_ic','t_stat','n']].sort_values(['horizon_m','variant']).to_string(index=False))
"
```

按市场状态查看（示例：plain）：

```bash
.\.venv\Scripts\python -c "import pandas as pd
s=pd.read_parquet('reports/pe_research_cn_all_open2open/ic_summary.parquet')
x=s[s.variant=='plain'][['horizon_m','regime','mean_ic','t_stat','n']].sort_values(['horizon_m','regime'])
print(x.to_string(index=False))
"
```

