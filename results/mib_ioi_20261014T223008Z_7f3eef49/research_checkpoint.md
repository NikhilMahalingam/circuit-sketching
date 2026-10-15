A. Validation mean |interaction| = 0.4830; relative scale = 15.2% (95% CI 14.5%--15.9%), above the prespecified 5% criterion. 64/66 signed-pair tests pass BY FDR 0.05.

B. Second-order MAE improvement over learned first order = 0.3688 (95% prompt CI 0.3015--0.3901); improvement supported.

C. MAE improvement by mask size: 1 heads: 0.6557, 2 heads: 0.4614, 3 heads: 0.5859, 6 heads: -0.0416, 9 heads: 0.3976. This descriptive pattern does not establish a monotonic trend.

D. Quadratic same-prompt-split MAE 0.3089; validation-prompt MAE 0.3680; change +0.0591. Coefficients were fitted on train prompts only.

E. 16 examples is the first tested count meeting the 0.90 rank threshold in every repeat; conditional on fixed EAP-selected heads and a full-train reference.

F. Exact pairs: 66 unique pairs across 2 prompt splits, 2112 actual forward batches, 16896 example-forwards, 158.58s wall; peak allocated CUDA 1134.3 MiB. Baseline/donor acquisition is separately charged in compute_accounting.csv.