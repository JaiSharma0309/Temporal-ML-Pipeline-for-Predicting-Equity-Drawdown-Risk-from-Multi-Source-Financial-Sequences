# Business Impact Summary

## Objective
Prioritize the riskiest names so a risk team can review fewer stocks while catching a large share of severe future drawdowns.

## Test-Set Impact
- Model: `random_forest`
- Base drawdown rate: `9.09%` (`22,019` events over `242,222` rows)
- Top 10% reviewed: `24,223` rows
- Precision in top 10%: `20.97%`
- Lift at top 10%: `2.31x`
- Share of all drawdowns captured by top 10%: `23.07%`

## Decile Contrast (Risk Buckets)
- Highest-risk decile event rate: `20.97%` (lift `2.31x`)
- Lowest-risk decile event rate: `3.08%` (lift `0.34x`)

## Threshold Tradeoff
- Selected threshold: `0.35`
- Precision: `15.63%`
- Recall: `57.62%`
- False positives / true positive: `5.40`

## Operational Recommendation
- Use score ranking as the primary workflow: review the top risk bucket first.
- If team capacity is fixed near 10%, the current model already gives meaningful enrichment versus random review.
- If you need a hard flag, use the selected threshold metrics above to tune precision/recall based on team tolerance for false alarms.

## Generated Artifacts
- `results/stage1/plots/04_test_decile_event_rate.png`
- `results/stage1/plots/05_test_capture_vs_workload.png`
- `results/stage1/reports/business_impact_summary.md`