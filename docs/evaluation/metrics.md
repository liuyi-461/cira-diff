# 指标登记表

| ID | 指标/家族 | 输入 | 状态 | 必须记录 |
| --- | --- | --- | --- | --- |
| MET-PIX-001 | ME/Bias | forecast、truth、mask | Not Implemented | sign convention、inverse normalization |
| MET-PIX-002 | MAE | forecast、truth、mask | Not Implemented | pixel weighting |
| MET-PIX-003 | RMSE/MSE | forecast、truth、mask | Not Implemented | lead-time aggregation |
| MET-STR-001 | SSIM | forecast、truth | Planned | window/range 与 BT applicability |
| MET-THR-001 | CSI/POD/FAR/ETS | thresholded forecast/truth | Planned | threshold provenance |
| MET-SPC-001 | energy spectrum | forecast、truth、spatial grid | Planned | detrending/windowing/binning |
| MET-OBJ-001 | cloud-object displacement/area | object masks | Planned | segmentation/connectivity |
| MET-TMP-001 | lead-time tendency error | forecast/truth sequences | Planned | temporal alignment |
| MET-PROB-001 | CRPS/reliability/spread-skill | ensemble、truth | Planned | members/seeds/calibration |

## 规则

只有在实现状态、公式、mask、threshold、单位、聚合方式和 artifact path 都有记录时，实验才能报告一个指标。
