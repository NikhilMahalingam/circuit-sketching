# Gate 3 sampling protocol — frozen before training

Frozen criteria SHA-256: `f200fc163bfb0e70b4790bde28ed3a6748bf9051c7d574ed7e23bda69cf0dbde`. Protocol SHA-256: `6ff6c0aed7cdfb8614bd16f022fcfbb8cdfb23d644691af8452a5dc4a9f9f808`.

**sampling**: lambda Uniform[1,N) + (1-lambda) sum_j q_j Uniform[L_j,H_j); q_j=w_j^1/sum(w_except_full); weights are linear CPR trapezoids.

**rounding**: c_j=int(p_j*N), positive floor matching the official evaluator; k remains floating, never rounded after jitter.

**jitter**: L=max(1,0.95*c); H=min(N-1,1.05*c); uniform over truncated interval, no clipping point masses.

**endpoint**: Full graph p=1 has fixed faithfulness 1, constant CPR contribution .25 and no ranking objective; exclude ONLY from grid training, renormalize .749 remaining mass. Evaluation grid unchanged. Uniform branch retains original near-full k<N behavior.

**rng**: Exactly one global CPU float32 torch.rand(1) per step, same consumption as official uniform. Inverse mixture CDF and category residual provide branch, grid index, jitter. Python row RNG untouched. No feedback/private RNG. Original checkpoint CPU/CUDA/Python/NumPy RNG restoration suffices.

**screen_rule**: Apply frozen primary numerical gates provisionally at 1000 vs matching 1000 baselines, OR frozen efficiency alternative vs 5000 baseline. This does not establish 5000 success. If none eligible stop. Rank eligible by mean paired relative CPR gain vs 1000; exact tie chooses lambda .5. No extra variants.

**confirmation**: Resume the one winner from step 1000 to 5000 for all three seeds; charge screen, restart/cache rebuilding and all other trials. Match frozen 5000 criteria without changes.

**memory**: Peak torch.cuda.max_memory_allocated, maximum across actual stages/attempts. Paired ratio of mean peaks, limit from original criteria.

**time**: CUDA-synchronized perf_counter. Screen training time/peak guards compare fresh uniform timing controls verified against Gate2; quality always Gate2. Confirmation training cost sums screen+continuation+donor-cache rebuilds against Gate2 measured 5000 training times; total costs additionally reported.

Static CPR-aware mixture; no online loss/validation adaptation. No edge-patching, optimizer, gradient or evaluator changes.
