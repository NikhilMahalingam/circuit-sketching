LIGHTWEIGHT SMOKE TEST — not a scientific conclusion.
1. Full second order held-out MAE 1.59237; first order 1.57344. Lower: first order (or tied).
2. Measured rank/MAE: [(4, 1.496809482574463)]; a rank trend requires at least two ranks.
3. No first-order disagreements on the measured held-out threshold sweep; difficult-decision benefit is unresolved.
4. Lowest measured held-out MAE: sketch_rank_4; improvement over first order 4.9%, construction time ratio 8.38x. Construction seconds (shared gradient allocated per method): {'exact_intervention': 0.46676229999866337, 'first_order': 0.13357479999831412, 'full_second_order': 3.718021199994837, 'sketch_rank_4': 1.1187235999968834}. This tradeoff does not establish end-to-end pruning savings.
Threshold coverage: no measured train effects lie in [0.01,0.30]; this edge/state set may not contain difficult pruning decisions in the requested range.
Intervention audit: maximum historical-adapter discrepancy from official MIB = 0.411971. The diagnostic adapter passed official endpoint checks.
