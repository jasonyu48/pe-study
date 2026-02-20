## PE study — backtest + IC report
- **Run dir**: `reports/pe_research_cn_all_open2open_qfq`
- **Best combo (by Sharpe)**: **h=1m**, **plain**

## All combinations (sorted by Sharpe)

| h(m) | variant          | Sharpe | IC(mean) | Return(total) | CAGR  | MaxDD  | AvgHold | N     |
| ---- | ---------------- | ------ | -------- | ------------- | ----- | ------ | ------- | ----- |
| 1    | plain            | 0.750  | 0.0488   | 174.8%        | 10.7% | -13.2% | 850     | 119.0 |
| 1    | industry_neutral | 0.697  | 0.0356   | 168.5%        | 10.5% | -16.0% | 846     | 119.0 |
| 24   | plain            | 0.683  | 0.0962   | 141.3%        | 9.3%  | -13.7% | 789     | 119.0 |
| 6    | plain            | 0.635  | 0.0509   | 130.2%        | 8.8%  | -15.4% | 836     | 119.0 |
| 12   | plain            | 0.633  | 0.0628   | 127.6%        | 8.6%  | -15.2% | 812     | 119.0 |
| 6    | industry_neutral | 0.620  | 0.0474   | 139.9%        | 9.2%  | -18.8% | 829     | 119.0 |
| 24   | industry_neutral | 0.594  | 0.0982   | 127.3%        | 8.6%  | -15.8% | 780     | 119.0 |
| 6    | threshold        | 0.572  | 0.0257   | 120.1%        | 8.3%  | -19.9% | 839     | 119.0 |
| 12   | industry_neutral | 0.569  | 0.0607   | 120.4%        | 8.3%  | -17.6% | 806     | 119.0 |
| 1    | threshold        | 0.560  | -0.0103  | 115.5%        | 8.0%  | -21.9% | 855     | 119.0 |
| 12   | threshold        | 0.530  | 0.0252   | 104.9%        | 7.5%  | -18.7% | 815     | 119.0 |
| 24   | threshold        | 0.501  | 0.0285   | 95.5%         | 7.0%  | -22.7% | 795     | 119.0 |

## Best combination — equity curve

![best equity curve](best_equity_curve.png)
