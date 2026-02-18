## PE study (A股估值因子研究)

本仓库聚焦一个研究任务：**P/E（TTM）因子的月度横截面预测能力**，并回答：

- **时间尺度**：用 1 / 6 / 12 / 24 个月作为前瞻收益时，IC 哪个最强？
- **市场状态**：在上涨/下跌、高/低波动环境下，IC 是否有差异？
- **因子改进**：普通 PE、加门槛的 PE、分行业的 PE，哪个更好？

### 目录结构（与 PE study 相关）

- `data/`: TuShare 下载脚本（不提交大数据文件）
- `src/pe_research.py`: 主研究脚本（生成 IC 汇总与时间序列）
- `src/tushare_client.py`: TuShare token 读取与 client 初始化
- `reports/pe_research_cn_all_open2open/README.md`: 一份示例结果解读（不提交 parquet 结果文件）

### 环境安装

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### TuShare Token

把 token 存在 `data/token`（已在 `.gitignore` 中忽略），然后可先自检权限：

```bash
python data/tushare_selftest.py --token-path data/token
```

---

## PE study 完整流程（open-to-open）

我们统一用一个口径：

- **信号更新**：月末（PE 快照是月末交易日）
- **前瞻收益**：**open-to-open**（用每个月第一个交易日开盘价 \(Open_t\) 计算 \(Open_{t+h}/Open_t-1\)）

### 1) 下载研究数据（全市场、月度）

1) 月末 PE（TTM）快照：

```bash
python data/fetch_cn_pe_monthly_tushare.py --start 2016-01-01 --end 2025-12-31 --out data/cn_pe_monthly.parquet
```

2) 月末价格（用于动量门槛与对齐）：

```bash
python data/fetch_cn_prices_monthend_tushare.py --start 2016-01-01 --end 2025-12-31 --out data/cn_prices_monthend.parquet
```

3) 月初开盘价（用于 open-to-open 前瞻收益）：

```bash
python data/fetch_cn_prices_monthstart_open_tushare.py --start 2016-01-01 --end 2025-12-31 --out data/cn_prices_monthstart_open.parquet
```

4) 行业映射（用于分行业 PE）：

```bash
python data/fetch_cn_stock_basic_industry_tushare.py --out data/cn_stock_industry.parquet
```

5) 市场状态（中证全指 000985.CSI，上涨/下跌 + 高/低波动）：

```bash
python data/fetch_cn_index_regimes_tushare.py --start 2016-01-01 --end 2025-12-31 --out data/cn_market_regimes.parquet
```

### 2) 运行研究脚本（生成 IC 结果）

```bash
python -m src.pe_research \
  --pe data/cn_pe_monthly.parquet \
  --prices data/cn_prices_monthend.parquet \
  --prices-monthstart-open data/cn_prices_monthstart_open.parquet \
  --industry data/cn_stock_industry.parquet \
  --regimes data/cn_market_regimes.parquet \
  --out-dir reports/pe_research_cn_all_open2open
```

输出：
- `reports/pe_research_cn_all_open2open/ic_series.parquet`
- `reports/pe_research_cn_all_open2open/ic_summary.parquet`

> 注意：`reports/**/*.parquet` 已被 `.gitignore` 忽略，不会提交到 GitHub。

