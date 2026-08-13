# Results


## Primary comparison (full 110k training split)

| run | model | l_max | train size | params | epochs | best epoch | val MAE (meV) | test MAE (meV) | minutes |
|---|---|---|---|---|---|---|---|---|---|
| baseline_f1_e200_s0 | baseline | — | 110,000 | 277,257 | 200 | 111 | 55.60 | **56.03** | 36.9 |
| tfn_l2_f1_e200_s0 | tfn | 2 | 110,000 | 572,681 | 200 | 184 | 58.60 | **59.43** | 276.6 |

## Ablation: maximum spherical-harmonic degree

| run | model | l_max | train size | params | epochs | best epoch | val MAE (meV) | test MAE (meV) | minutes |
|---|---|---|---|---|---|---|---|---|---|
| tfn_l0_f1_e50_s0 | tfn | 0 | 110,000 | 154,505 | 50 | 49 | 113.43 | **113.59** | 22.9 |
| tfn_l1_f1_e50_s0 | tfn | 1 | 110,000 | 306,249 | 50 | 49 | 90.58 | **89.55** | 34.2 |
| tfn_l2_f1_e50_s0 | tfn | 2 | 110,000 | 572,681 | 50 | 49 | 76.84 | **78.63** | 84.5 |

## Data efficiency

| run | model | l_max | train size | params | epochs | best epoch | val MAE (meV) | test MAE (meV) | minutes |
|---|---|---|---|---|---|---|---|---|---|
| baseline_f0.1_e250_s0 | baseline | — | 11,000 | 277,257 | 250 | 135 | 141.17 | **142.03** | 5.9 |
| baseline_f0.25_e200_s0 | baseline | — | 27,500 | 277,257 | 200 | 109 | 97.06 | **98.06** | 9.4 |
| baseline_f0.5_e100_s0 | baseline | — | 55,000 | 277,257 | 100 | 75 | 77.68 | **78.06** | 12.3 |
| baseline_f1_e50_s0 | baseline | — | 110,000 | 277,257 | 50 | 48 | 63.75 | **63.82** | 12.3 |
| tfn_l2_f0.1_e250_s0 | tfn | 2 | 11,000 | 572,681 | 250 | 165 | 184.29 | **182.66** | 36.4 |
| tfn_l2_f0.25_e200_s0 | tfn | 2 | 27,500 | 572,681 | 200 | 127 | 115.43 | **117.31** | 62.4 |
| tfn_l2_f0.5_e100_s0 | tfn | 2 | 55,000 | 572,681 | 100 | 99 | 90.87 | **90.70** | 77.4 |
| tfn_l2_f1_e50_s0 | tfn | 2 | 110,000 | 572,681 | 50 | 49 | 76.84 | **78.63** | 84.5 |
