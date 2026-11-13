"""Discovery-only references, streamed to disk and read one sampled row at a time."""

import time
from pathlib import Path
from hashlib import file_digest

import numpy as np
import torch

from src.mattr_fidelity_splice.core import read
from src.mattr_fidelity_splice.inference import Recorder, model_identity
from src.mattr_matched_size.topology import graph_new
from src.mattr_optimization.common import sha, write_json
from .preflight import dataset

BATCH = 20
VERSION = 1


def disk_hash(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def verified_targets(folder, identity=None, *, full=True):
    path = folder / "completed.json"
    if not path.exists():
        return False
    record = read(path)
    if identity is not None and record["identity"] != identity:
        raise ValueError("Target cache identity mismatch")
    for name, details in record["files"].items():
        file = folder / name
        stat = file.stat()
        if stat.st_size != details["bytes"] or (not full and stat.st_mtime_ns != details["mtime_ns"]):
            raise ValueError(f"Target cache changed: {name}")
        if full and disk_hash(file) != details["sha256"]:
            raise ValueError(f"Target cache hash mismatch: {name}")
    return True


@torch.no_grad()
def build(model, frame, folder, meter, *, manifest_path=None, validate_seed=6100):
    from eap.evaluate import evaluate_baseline, evaluate_graph
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    from MIB_circuit_track.metrics import get_metric
    from functools import partial

    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path or folder.parent / "discovery.csv"
    identity = dict(model=model_identity(model), implementation_version=VERSION, batch=BATCH,
                    manifest_sha256=sha(manifest_path),
                    exact_rows=frame.official_filtered_index.astype(int).tolist())
    if verified_targets(folder, identity):
        return read(folder / "metadata.json")

    ds = dataset(model, frame, "train")
    loader = ds.to_dataloader(BATCH)
    graph = graph_new()
    graph.apply_topn(0, absolute=False, prune=True)
    metric = partial(get_metric("logit_diff", "ioi", model.tokenizer, model), mean=False, loss=False)
    path = folder / "logp.npy"
    partial_path = folder / "logp.partial.npy"
    mmap = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32,
                                    shape=(len(frame), model.cfg.d_vocab))
    rec = Recorder(model, frame)

    def cache_metric(logits, donor, lengths, labels):
        from MIB_circuit_track.metrics import get_logit_positions
        offset = rec.cursor
        result = rec(logits, donor, lengths, labels)
        logp = get_logit_positions(logits, lengths).float().log_softmax(-1)
        torch.testing.assert_close(logp.logsumexp(-1), torch.zeros_like(result), atol=3e-6, rtol=0)
        mmap[offset:rec.cursor] = logp.cpu().numpy()
        return result

    with meter.stage("discovery_clean_targets"):
        full = evaluate_baseline(model, loader, cache_metric, quiet=True)
        np.testing.assert_array_equal(full.numpy(), rec.arrays()["margin"])
    with meter.stage("discovery_empty_targets"):
        empty = evaluate_graph(model, graph, loader, metric, quiet=True, intervention="patching")
    denominator = full.mean().item() - empty.mean().item()
    assert np.isfinite(denominator) and abs(denominator) > 1e-8
    mmap.flush()
    rows = np.random.default_rng(validate_seed).choice(len(frame), size=50, replace=False)
    check_frame = frame.iloc[rows].reset_index(drop=True)
    subset = dataset(model, check_frame, "train")
    check = Recorder(model, check_frame, save_logp=True)
    with meter.stage("independent_target_verification"):
        evaluate_baseline(model, subset.to_dataloader(1), check, quiet=True)
        independent = check.arrays()
        np.testing.assert_allclose(independent["margin"], full.numpy()[rows], atol=1e-4, rtol=2e-5)
        np.testing.assert_allclose(independent["logp"], mmap[rows], atol=1e-4, rtol=2e-5)
    # Replay independently computed official references to verify the exact normalization path.
    from unittest.mock import patch
    import MIB_circuit_track.evaluation as official
    refs = {}
    with patch.object(official, "evaluate_baseline", return_value=full), \
         patch.object(official, "evaluate_graph", return_value=empty):
        area = evaluate_area_under_curve(model, graph, loader, metric, quiet=True,
                                        percentages=(.1,), refs=refs)
    assert refs["baseline"] - refs["corrupted"] == denominator and area[4] == [0.]
    del mmap
    partial_path.replace(path)
    np.save(folder / "margin.npy", full.numpy(), allow_pickle=False)
    np.save(folder / "empty.npy", empty.numpy(), allow_pickle=False)
    result = dict(passed=True, strategy="float32 CPU memory-mapped cache; transfer sampled row only",
                  D=denominator, clean_mean=refs["baseline"], empty_mean=refs["corrupted"],
                  validated_rows=rows.tolist(), count=len(frame), full_vocab=model.cfg.d_vocab,
                  bytes=path.stat().st_size, target_identity=identity)
    write_json(folder / "metadata.json", result)
    files = {}
    for file in (path, folder / "margin.npy", folder / "empty.npy", folder / "metadata.json"):
        stat = file.stat()
        files[file.name] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=disk_hash(file))
    write_json(folder / "completed.json", dict(identity=identity, files=files))
    return result


class Targets:
    def __init__(self, folder, loss):
        keys = {"margin": ("margin",), "kl": ("logp",), "repo_cmd": ("margin", "empty")}[loss]
        self.arrays = {key: np.load(Path(folder) / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                       for key in keys}
        self.read_seconds = self.transfer_seconds = 0.
        self.reads = 0

    def row(self, index, device):
        start = time.perf_counter()
        values = {k: torch.from_numpy(np.array(v[index:index + 1], copy=True)) for k, v in self.arrays.items()}
        transfer = time.perf_counter()
        result = {k: v.to(device) for k, v in values.items()}
        self.read_seconds += transfer - start
        self.transfer_seconds += time.perf_counter() - transfer
        self.reads += 1
        return result
