# Training schedule audit

The pinned trainer has constant learning rate and temperature; planned duration only controls the loop bound. A real CPU execution of the official trainer verified that update 1,000 of a 5,000-update run equals a standalone 1,000-update run, including Adam state. This is a synthetic correctness test, not an IOI research measurement.

Official source: `/mnt/h/GenAI Research/sketch_prune/external/matryoshka-attribution/src/matryoshka_attribution/trainer.py`, lines 35–195, SHA-256 `b987d4b174e6147613d8f806ba41ee65ce0fde8214cbf568f863ce1c701ab7e9`.

Thirty-update CPU tests checked both baseline and H2 exact 15+15 restoration. The model loss and activation patching are not reimplemented. Training-example draws and uniform-k draws are separate frozen streams; hierarchy initialization is deterministic zero. Historical42–44 k sequences are verified for all5,000 draws;45–46 use independent explicitly seeded CPU generators. Intermediate checkpoints are reusable only when data, source, scores, optimizer and full draw streams pass the separate inventory.
