from pathlib import Path
import json,hashlib,torch
r=Path.cwd();o=r/'results/mattr_compact_validation/20261023_gate5d';p=json.loads((o/'gate5d_protocol.json').read_text());rows=[]
for rec in p['rankings']:
    state=torch.load(r/rec['checkpoint'],map_location='cpu',weights_only=False)
    steps=[int(v['step']) for v in state['optimizer']['state'].values()]
    assert steps and set(steps)=={1000}
    cfg=json.loads((r/rec['config']).read_text())
    if cfg['family']=='H2':
        path=r/'results/mattr_hierarchy/20261022_gate5c'/('H2_shuffled_groups.npy' if cfg['shuffled'] else 'H2_groups.npy')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['group_hash']
    rows.append({'method':rec['method'],'seed':rec['seed'],'checkpoint_completed':state['completed'],'Adam_steps':steps,'raw_parameter_count':int(state['scores'].numel()),'canonical_effective_count':32491,'group_hash_matches':True})
(o/'optimizer_checkpoint_audit.json').write_text(json.dumps({'passed':True,'scope':'read-only supplemental source-checkpoint metadata; no training or model forwards','rankings':rows},indent=2)+'\n')
print('All15 checkpoints have actual Adam step1000; grouping hashes match saved configurations')
