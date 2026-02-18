## PE study — backtest + IC report
- **Run dir**: `reports/pe_research_cn_all_open2open_final_zombiefix`
- **Best combo (by Sharpe)**: **h=1m**, **plain**

## All combinations (sorted by Sharpe)

| h(m) | variant          | Sharpe | IC(mean) | Return(total) | CAGR | MaxDD  | AvgHold | N     |
| ---- | ---------------- | ------ | -------- | ------------- | ---- | ------ | ------- | ----- |
| 1    | plain            | 0.603  | 0.0372   | 118.7%        | 8.2% | -13.7% | 850     | 119.0 |
| 24   | plain            | 0.554  | 0.0439   | 98.6%         | 7.2% | -14.2% | 789     | 119.0 |
| 1    | industry_neutral | 0.542  | 0.0238   | 108.0%        | 7.7% | -19.4% | 846     | 119.0 |
| 12   | plain            | 0.488  | 0.0189   | 83.3%         | 6.3% | -18.2% | 812     | 119.0 |
| 6    | plain            | 0.487  | 0.0221   | 84.3%         | 6.4% | -19.2% | 836     | 119.0 |
| 6    | industry_neutral | 0.474  | 0.0163   | 89.0%         | 6.6% | -25.0% | 829     | 119.0 |
| 24   | industry_neutral | 0.461  | 0.0408   | 82.9%         | 6.3% | -20.6% | 780     | 119.0 |
| 12   | industry_neutral | 0.424  | 0.0134   | 74.0%         | 5.7% | -23.5% | 806     | 119.0 |
| 6    | threshold        | 0.407  | 0.0006   | 70.2%         | 5.5% | -26.0% | 840     | 119.0 |
| 12   | threshold        | 0.397  | -0.0037  | 65.9%         | 5.2% | -23.5% | 815     | 119.0 |
| 24   | threshold        | 0.381  | -0.0098  | 61.4%         | 4.9% | -26.9% | 795     | 119.0 |
| 1    | threshold        | 0.380  | -0.0193  | 61.9%         | 5.0% | -27.3% | 857     | 119.0 |

## Best combination — equity curve

![best equity curve](best_equity_curve.png)
