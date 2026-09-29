# TPBench seed extension results

Dialogue bootstrap: 10,000 replicates; random selector seeds: 42, 43, 44, 45, 46

## RiSAWOZ P2 strict

| Method | Mean | 95% CI | n |
|---|---:|---:|---:|
| full_context | 0.232 | [0.174, 0.290] | 207 |
| recency | 0.155 | [0.106, 0.208] | 207 |
| first_n | 0.029 | [0.010, 0.053] | 207 |
| uniform_stride | 0.121 | [0.077, 0.164] | 207 |
| random_selection | 0.150 | [0.120, 0.183] | 207 |

## LongMemEval-KU P2 strict

| Method | Mean | 95% CI | n |
|---|---:|---:|---:|
| full_context | 0.417 | [0.306, 0.528] | 72 |
| recency | 0.319 | [0.222, 0.431] | 72 |
| first_n | 0.097 | [0.041, 0.167] | 72 |
| uniform_stride | 0.250 | [0.153, 0.347] | 72 |
| random_selection | 0.228 | [0.164, 0.294] | 72 |

LongMemEval-KU reports current-information retention (P2).
