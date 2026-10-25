"""Immutable identities, code lock and integrity-checked, resumable artifacts."""
import contextlib
import hashlib
import json
from pathlib import Path
import time
import uuid

from src.mattr_optimization.common import ROOT, GRID, MIB, configure, git, provenance, sha, write_json

DEFAULT = ROOT / 'results/mattr_fidelity_splice/gate5g'
HISTORY = ROOT / 'results/mattr_data_scaling/gate5f'
CAPS = [60, 80, 100, 116, 140, 180, 240, 296, 400, 640]
SEEDS = list(range(42, 47))
SIZES = [324, 1624]
N = 32491


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def marker(folder, identity, paths):
    folder = Path(folder)
    write_json(folder / 'completed.json', dict(identity=identity, passed=True,
        sha256={Path(p).relative_to(folder).as_posix(): sha(p) for p in paths}))


def verified(folder, identity=None):
    path = Path(folder) / 'completed.json'
    if not path.exists():
        return False
    record = read(path)
    if not record['passed'] or (identity is not None and record['identity'] != identity):
        raise ValueError(f'Completed identity mismatch: {folder}')
    for name, value in record['sha256'].items():
        if sha(Path(folder) / name) != value:
            raise ValueError(f'Completed output hash mismatch: {folder}/{name}')
    return True


def check_files(mapping, base=ROOT):
    for name, value in mapping.items():
        if sha(base / name) != value:
            raise ValueError('Frozen input/code hash mismatch: ' + name)


def pinned_extra_sources():
    """Pin all imported EAP/MIB Python, including utils/tokenization/hook creation.

    Historical Windows checkouts can differ from Git blobs in CRLF only. Verify
    normalized bytes against pinned HEAD, then freeze actual bytes without editing.
    Tracked upstream pycache files are runtime products, never experiment sources.
    """
    result={}
    for repo,folder in [(MIB,MIB/'MIB_circuit_track'),(MIB/'EAP-IG',MIB/'EAP-IG/src/eap')]:
        for path in folder.glob('*.py'):
            original=git(repo,'show','HEAD:'+path.relative_to(repo).as_posix())
            if path.read_bytes().replace(b'\r\n',b'\n')!=original.replace(b'\r\n',b'\n'):
                raise ValueError('Pinned upstream source has uncommitted code changes: '+str(path))
            result[relative(path)]=sha(path)
    return result


def protocol(out, live=False):
    path = out / 'gate5g_protocol.json'
    if sha(path) != (out / 'gate5g_protocol.sha256').read_text().strip():
        raise ValueError('Frozen protocol hash mismatch')
    p = read(path)
    if digest(p) != sha(path):
        raise ValueError('Protocol is not canonical JSON')
    check_files(p['sources'])
    check_files(p['artifacts'], out)
    for rec in p['rankings']:
        check_files(rec['input_hashes'])
    if live:
        if git(ROOT, 'rev-parse', 'HEAD').decode().strip() != p['implementation_commit']:
            raise ValueError('HEAD differs from frozen implementation commit; no override')
        tracked = list(p['sources'])
        if git(ROOT, 'status', '--porcelain', '--', *tracked).strip():
            raise ValueError('Tracked experiment sources are dirty')
        if provenance() != p['dependencies']:
            raise ValueError('Pinned runtime/dependencies differ from frozen protocol')
    return p


def identity(out, p, job):
    return dict(protocol_sha256=sha(out / 'gate5g_protocol.json'),
        implementation_commit=p['implementation_commit'], job=job,
        cohort_sha256=p['cohort_sha256'], model_sha256=p['model']['weight_sha256'])


@contextlib.contextmanager
def attempt(out, key):
    import fcntl
    locks = out / 'locks'
    locks.mkdir(exist_ok=True)
    work = out / 'attempts' / key / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
    work.mkdir(parents=True)
    with (locks / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = dict(key=key, status='running', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        write_json(work / 'attempt.json', record)
        start = time.perf_counter()
        try:
            yield work
            record['status'] = 'complete'
        except BaseException as exc:
            record.update(status='failed_or_interrupted', error=repr(exc), advice='Stop on correctness/OOM. Do not change registered settings. Resume identical jobs after interruptions.')
            raise
        finally:
            record['seconds'] = time.perf_counter() - start
            write_json(work / 'attempt.json', record)


class ArrayCache:
    """All-or-nothing byte-verified arrays. Incomplete writes are quarantined."""
    def __init__(self, folder, context):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.context = context
        self.hits = self.misses = 0

    def get(self, key, compute):
        import numpy as np
        ident = dict(context=self.context, key=key)
        path = self.folder / (digest(ident) + '.npz')
        meta = path.with_suffix('.json')
        if path.exists() and not meta.exists():
            quarantine = self.folder / 'incomplete'
            quarantine.mkdir(exist_ok=True)
            path.replace(quarantine / (path.name + '.' + uuid.uuid4().hex))
        if path.exists():
            saved = read(meta)
            if saved['identity'] != ident or saved['sha256'] != sha(path):
                raise ValueError('Cache identity/hash failure; refuse unsafe reuse')
            self.hits += 1
            with np.load(path, allow_pickle=False) as data:
                return {k: data[k].copy() for k in data.files}
        arrays = compute()
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Nonfinite cached output')
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        temp.replace(path)
        write_json(meta, dict(identity=ident, sha256=sha(path)))
        self.misses += 1
        return {k: v.copy() for k, v in arrays.items()}


def runtime_guard(out, p, approved=False):
    """Calibrate once on a complete Part A job, then stop before further jobs."""
    records = []
    for job in p['jobs']:
        folder = out / 'evaluations' / job['id']
        if verified(folder, identity(out, p, job)):
            records.append((job, read(folder / 'summary.json')))
    calibrated = [r for j, r in records if j['part'] == 'A']
    elapsed = sum(read(x).get('seconds', 0) for x in (out / 'attempts').rglob('attempt.json'))
    elapsed += read(out / 'preflight_cpu.json').get('seconds',0.)
    completed = {j['id'] for j, _ in records}
    remaining = [j for j in p['jobs'] if j['id'] not in completed]
    calibrated_rates=[]
    for job,result in records:
        if job['part']!='A':continue
        attempts=[read(x) for x in (out/'attempts'/job['id']).glob('*/attempt.json')]
        elapsed_job=max([a['seconds'] for a in attempts if a['status']=='complete'] or [result['evaluation_seconds']])
        calibrated_rates.append(elapsed_job/max(1,result['requested_masks']))
    seconds_per_query = max(calibrated_rates or
                            [p['compute_projection']['seconds_per_query']])
    projected = elapsed + sum(j['query_count'] for j in remaining) * seconds_per_query
    record = dict(projected_cumulative_seconds=projected, observed_attempt_seconds=elapsed,
        remaining_jobs=len(remaining), seconds_per_query=seconds_per_query,
        calibrated=bool(calibrated), limit_seconds=18000, explicit_approval=approved)
    write_json(out / 'runtime_projection.json', record)
    if approved:
        write_json(out / 'manual_runtime_approval.json', dict(protocol_sha256=sha(out / 'gate5g_protocol.json'), **record))
    if calibrated and projected > 18000 and not approved:
        raise RuntimeError('Projection exceeds five hours. Review runtime_projection.json and explicitly pass --approve-over-5h. No settings reduced.')
    return record
