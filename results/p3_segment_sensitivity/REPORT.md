# Primary P3 segment scoring and whole-field sensitivity

Both scorers are recomputed from the saved reader outputs. Segment-local scores are primary; original whole-field scores are a sensitivity analysis.

Unparseable answers receive zero under primary segment-local scoring. Format loss and parsed cross-segment loss are reported separately.

| Reader | Dataset | Method | Original loose / strict | Segment-local loose / strict | Parse rate |
|---|---|---|---|---|---|
| chunkkv | multiwoz | chunkkv_r03_kvpress | 0.130 / 0.100 | 0.100 / 0.085 | 0.875 |
| chunkkv | sgd | chunkkv_r03_kvpress | 0.175 / 0.160 | 0.155 / 0.140 | 0.865 |
| llama | multiwoz | attention_h2o_cache | 0.145 / 0.097 | 0.068 / 0.048 | 0.725 |
| llama | multiwoz | embedding_mmr_cache | 0.140 / 0.110 | 0.120 / 0.110 | 0.750 |
| llama | multiwoz | first_n | 0.132 / 0.085 | 0.058 / 0.048 | 0.835 |
| llama | multiwoz | full_context | 0.537 / 0.497 | 0.505 / 0.475 | 0.945 |
| llama | multiwoz | llmlingua2_cache | 0.165 / 0.147 | 0.117 / 0.110 | 0.805 |
| llama | multiwoz | random_seed42 | 0.192 / 0.148 | 0.145 / 0.128 | 0.907 |
| llama | multiwoz | recency | 0.068 / 0.058 | 0.040 / 0.040 | 0.885 |
| llama | multiwoz | uniform_stride | 0.285 / 0.222 | 0.220 / 0.178 | 0.967 |
| llama | sgd | attention_h2o_cache | 0.322 / 0.183 | 0.288 / 0.168 | 0.908 |
| llama | sgd | embedding_mmr_cache | 0.243 / 0.157 | 0.185 / 0.128 | 0.762 |
| llama | sgd | first_n | 0.330 / 0.172 | 0.313 / 0.160 | 0.953 |
| llama | sgd | full_context | 0.742 / 0.658 | 0.728 / 0.652 | 0.977 |
| llama | sgd | llmlingua2_cache | 0.135 / 0.072 | 0.117 / 0.060 | 0.860 |
| llama | sgd | random_seed42 | 0.275 / 0.182 | 0.238 / 0.162 | 0.967 |
| llama | sgd | recency | 0.083 / 0.075 | 0.067 / 0.065 | 0.870 |
| llama | sgd | uniform_stride | 0.452 / 0.298 | 0.432 / 0.288 | 0.973 |
| mistral | multiwoz | attention_h2o_cache | 0.097 / 0.068 | 0.045 / 0.032 | 0.720 |
| mistral | multiwoz | embedding_mmr_cache | 0.122 / 0.093 | 0.060 / 0.048 | 0.798 |
| mistral | multiwoz | first_n | 0.070 / 0.050 | 0.018 / 0.012 | 0.612 |
| mistral | multiwoz | full_context | 0.258 / 0.238 | 0.158 / 0.142 | 0.583 |
| mistral | multiwoz | llmlingua2_cache | 0.138 / 0.127 | 0.085 / 0.078 | 0.767 |
| mistral | multiwoz | random_seed42 | 0.118 / 0.093 | 0.070 / 0.060 | 0.743 |
| mistral | multiwoz | recency | 0.068 / 0.057 | 0.025 / 0.020 | 0.852 |
| mistral | multiwoz | uniform_stride | 0.157 / 0.110 | 0.082 / 0.058 | 0.608 |
| mistral | sgd | attention_h2o_cache | 0.285 / 0.155 | 0.183 / 0.107 | 0.783 |
| mistral | sgd | embedding_mmr_cache | 0.277 / 0.177 | 0.178 / 0.117 | 0.768 |
| mistral | sgd | first_n | 0.253 / 0.125 | 0.180 / 0.073 | 0.647 |
| mistral | sgd | full_context | 0.408 / 0.355 | 0.342 / 0.303 | 0.673 |
| mistral | sgd | llmlingua2_cache | 0.125 / 0.080 | 0.068 / 0.048 | 0.782 |
| mistral | sgd | random_seed42 | 0.198 / 0.135 | 0.150 / 0.103 | 0.790 |
| mistral | sgd | recency | 0.063 / 0.050 | 0.033 / 0.033 | 0.793 |
| mistral | sgd | uniform_stride | 0.303 / 0.202 | 0.258 / 0.173 | 0.750 |
