| rq   | model   | contrast                                    | primary_metric   |   p_primary |   p_holm |   delta | ci                 |
|:-----|:--------|:--------------------------------------------|:-----------------|------------:|---------:|--------:|:-------------------|
| RQ1  | cat     | F1 vs F0 (no CS, raw)                       | pr_auc           |      0.0280 |   0.2239 |  0.0019 | [+0.0003, +0.0036] |
| RQ2  | cat     | weight vs none (F0, raw)                    | f2@op            |      0.3168 |   1.0000 |  0.0006 | [-0.0006, +0.0019] |
| RQ3  | cat     | best calibrator vs raw (F1, weight)         | brier            |      0.0010 |   0.0180 | -0.0412 | [-0.0418, -0.0407] |
| ALL  | cat     | F1+weight+calibrated@tf2 vs F0+none+raw@0.5 | f2@op            |      0.0010 |   0.0060 |  0.4208 | [+0.4154, +0.4265] |
| RQ1  | dt      | F1 vs F0 (no CS, raw)                       | pr_auc           |      0.0010 |   0.0180 |  0.0165 | [+0.0125, +0.0206] |
| RQ2  | dt      | weight vs none (F0, raw)                    | f2@op            |      0.9075 |   1.0000 | -0.0002 | [-0.0021, +0.0020] |
| RQ3  | dt      | best calibrator vs raw (F1, none)           | brier            |      0.0010 |   0.0180 | -0.0004 | [-0.0004, -0.0003] |
| ALL  | dt      | F1+none+calibrated@tf2 vs F0+none+raw@0.5   | f2@op            |      0.0010 |   0.0060 |  0.4532 | [+0.4480, +0.4585] |
| RQ1  | lgbm    | F1 vs F0 (no CS, raw)                       | pr_auc           |      0.0790 |   0.5527 |  0.0014 | [-0.0001, +0.0029] |
| RQ2  | lgbm    | weight vs none (F0, raw)                    | f2@op            |      0.6327 |   1.0000 | -0.0003 | [-0.0014, +0.0008] |
| RQ3  | lgbm    | best calibrator vs raw (F1, none)           | brier            |      0.0010 |   0.0180 | -0.0001 | [-0.0001, -0.0001] |
| ALL  | lgbm    | F1+none+calibrated@tf2 vs F0+none+raw@0.5   | f2@op            |      0.0010 |   0.0060 |  0.4296 | [+0.4242, +0.4354] |
| RQ1  | lr      | F1 vs F0 (no CS, raw)                       | pr_auc           |      0.0010 |   0.0180 |  0.0157 | [+0.0131, +0.0183] |
| RQ2  | lr      | weight vs none (F0, raw)                    | f2@op            |      0.2919 |   1.0000 |  0.0003 | [-0.0003, +0.0009] |
| RQ3  | lr      | best calibrator vs raw (F1, rus)            | brier            |      0.0010 |   0.0180 | -0.0443 | [-0.0449, -0.0438] |
| ALL  | lr      | F1+rus+calibrated@tf2 vs F0+none+raw@0.5    | f2@op            |      0.0010 |   0.0060 |  0.4578 | [+0.4527, +0.4631] |
| RQ1  | rf      | F1 vs F0 (no CS, raw)                       | pr_auc           |      0.0010 |   0.0180 |  0.0089 | [+0.0067, +0.0111] |
| RQ2  | rf      | weight vs none (F0, raw)                    | f2@op            |      0.1449 |   0.8696 |  0.0009 | [-0.0004, +0.0022] |
| RQ3  | rf      | best calibrator vs raw (F1, weight)         | brier            |      0.0010 |   0.0180 | -0.0376 | [-0.0380, -0.0371] |
| ALL  | rf      | F1+weight+calibrated@tf2 vs F0+none+raw@0.5 | f2@op            |      0.0010 |   0.0060 |  0.4915 | [+0.4864, +0.4966] |
| RQ1  | xgb     | F1 vs F0 (no CS, raw)                       | pr_auc           |      0.0010 |   0.0180 |  0.0037 | [+0.0019, +0.0056] |
| RQ2  | xgb     | weight vs none (F0, raw)                    | f2@op            |      0.4418 |   1.0000 |  0.0004 | [-0.0006, +0.0014] |
| RQ3  | xgb     | best calibrator vs raw (F1, none)           | brier            |      0.0010 |   0.0180 | -0.0001 | [-0.0001, -0.0001] |
| ALL  | xgb     | F1+none+calibrated@tf2 vs F0+none+raw@0.5   | f2@op            |      0.0010 |   0.0060 |  0.4328 | [+0.4272, +0.4383] |