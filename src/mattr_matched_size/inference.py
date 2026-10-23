"""Inference only; mandatory first-ranking two-hour projection gate."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import ROOT, configure, provenance, sha, write_json
from src.mattr_optimization.gate1 import LiveAccounting
from src.mattr_compact_validation.inference import correctness, Recorder, assign_scores
from src.mattr_hierarchy.runner import weight_hash
from .design import protocol
from .evaluation import OfficialEngine


def cache_correctness(model,graph,ds,p,out,meter):
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    small = copy.copy(ds)
    small.dataset = ds.dataset.select(range(2))
    scores = torch.load(ROOT/p['rankings'][0]['scores'],map_location='cpu',weights_only=False)
    assign_scores(graph,scores)
    rec = Recorder(model)
    with meter.stage('exact_output_cache_correctness'),torch.no_grad():
        raw = evaluate_area_under_curve(model,graph,small.to_dataloader(2),rec,level='edge',absolute=False,quiet=True,refs={})
        plain_margin,plain_top1 = rec.arrays(2)
        engine = OfficialEngine(model,graph,small,out/'engine_audit',sha(out/'gate5e_protocol.json'),'first2_frozen_prompts',batch=2)
        result,curve,per = engine.area(scores,p['grid'])
        for a,b in zip(raw,result):
            np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-6)
        np.testing.assert_array_equal(per.pivot(index='fraction',columns='example',values='margin').loc[p['grid']].to_numpy(),plain_margin[2:])
        np.testing.assert_array_equal(per.pivot(index='fraction',columns='example',values='top1').loc[p['grid']].to_numpy(),plain_top1[2:])
        before = meter.counts.copy()
        again,_,_ = engine.area(scores,p['grid'])
        assert before == meter.counts,'Exact-cache hits executed real model forwards.'
        assert raw == again
    write_json(out/'exact_cache_correctness.json',{'passed':True,'official_uncached_equals_cached':True,
               'per_prompt_margin_and_top1_equal':True,'repeat_forwards':0,'cache_hits':engine.cache.hits,
               'masked_activations_cached':False,'scope':'same2 frozen examples, unchanged official patcher and integrals'})


def run(out):
    configure()
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    from MIB_circuit_track.dataset import HFEAPDataset
    from eap.graph import Graph
    from src.mattr_optimization.gate4_init import edge_mapping
    p = protocol(out)
    if (out/'inference_process_summary.json').exists() or (out/'forward_events.jsonl').exists():
        raise ValueError('Existing attempt preserved. Do not overwrite accounting or rerun automatically.')
    assert torch.cuda.is_available()
    assert provenance() == p['source_provenance']
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meter = LiveAccounting(out)
    start = time.perf_counter()
    success = stopped = False
    engine = None
    try:
        with meter.stage('model_loading'),torch.no_grad():
            model = meter.attach(HookedTransformer.from_pretrained(p['model']))
            model.eval()
            for k,v in p['model_config'].items():
                if k in ('n_layers','n_heads','d_model'):
                    assert getattr(model.cfg,k) == v
                else:
                    setattr(model.cfg,k,v)
            assert model.cfg.dtype == torch.float32
            before = weight_hash(model)
            assert all(r['model_weights_sha256'] == before for r in p['rankings'])
        graph = Graph.from_model(model)
        pd.testing.assert_frame_equal(edge_mapping(graph,graph),pd.read_csv(out/'edge_mapping.csv',keep_default_na=False))
        with meter.stage('cohort_loading'):
            ds = HFEAPDataset('mib-bench/ioi',AutoTokenizer.from_pretrained('gpt2'),split='validation',task='ioi',model_name='gpt2')
            manifest = pd.read_csv(out/'cohort_manifest.csv',keep_default_na=False)
            ds.dataset = ds.dataset.select(manifest.official_filtered_index.astype(int).tolist())
            for i,row in manifest.iterrows():
                clean,donor,labels = ds[i]
                assert clean == row.clean_prompt and donor == row.corrupted_prompt
                assert labels == [row.correct_token,row.incorrect_token]
        correctness(model,graph,ds,p,out,meter)
        cache_correctness(model,graph,ds,p,out,meter)
        engine = OfficialEngine(model,graph,ds,out,sha(out/'gate5e_protocol.json'),p['cohort']['manifest_sha256'],batch=p['batch_size'])
        print(f'Gate5E: {len(p["catalog"])} full-grid rankings plus {len(pd.read_csv(out/"circuit_queries.csv"))} cap/upper queries; '
              f'{len(ds)} prompts, batch20, {p["budget"]["planned_circuit_requests"]} requested circuits before exact-output reuse.',flush=True)
        for j,ranking in enumerate(p['catalog'],1):
            key = ranking['key']
            scores = torch.load(ROOT/ranking['scores'],map_location='cpu',weights_only=False)
            dest = out/'nominal'/key
            dest.mkdir(parents=True,exist_ok=True)
            with meter.stage('nominal_'+key),torch.no_grad():
                result,curve,per = engine.area(scores,p['grid'])
            curve.to_csv(dest/'curve.csv',index=False)
            per.to_csv(dest/'per_prompt.csv',index=False)
            write_json(dest/'official_result.json',{'protocol_sha256':sha(out/'gate5e_protocol.json'),
                       'ranking':ranking,'result':result,'per_prompt_sha256':sha(dest/'per_prompt.csv')})
            print(f'Nominal {j}/{len(p["catalog"])} {key}: CPR={result[1]:.6f}, CMD={result[2]:.6f}, cache_hits={engine.cache.hits}',flush=True)
            meter.status(stage_group='nominal',completed=j,total=len(p['catalog']),ranking=key,exact_cache_hits=engine.cache.hits)
            if j == 1:
                seconds = meter.rows[-1]['seconds']
                projected = seconds/10*p['budget']['planned_circuit_requests']
                projection = {'first_ranking_seconds':seconds,'requested_circuits':p['budget']['planned_circuit_requests'],
                              'projected_seconds':projected,'threshold_seconds':7200,
                              'includes_first_reference_overhead':True,'passed':projected <= 7200}
                write_json(out/'first_ranking_projection.json',projection)
                print(json.dumps(projection),flush=True)
                if not projection['passed']:
                    stopped = True
                    break
        if not stopped:
            queries = pd.read_csv(out/'circuit_queries.csv')
            allowed = {r['key']:r for r in p['catalog'] if r['kind'] != 'closure'}
            assert set(queries.ranking) == set(allowed),'Closure matched-size query prohibited.'
            for j,(key,group) in enumerate(queries.groupby('ranking',sort=False),1):
                ranking = allowed[key]
                scores = torch.load(ROOT/ranking['scores'],map_location='cpu',weights_only=False)
                dest = out/'matched'/key
                dest.mkdir(parents=True,exist_ok=True)
                values,per = [],[]
                with meter.stage('matched_'+key),torch.no_grad():
                    for query in group.itertuples():
                        measured,margin,top1 = engine.selected(scores,query.nominal_n,query.mask_sha256)
                        assert measured['actual_edges'] == query.actual_edges
                        values.append({**query._asdict(),**measured})
                        for i in range(len(ds)):
                            per.append({'selection':query.selection,'target':query.target,'example':i,
                                        'margin':float(margin[i]),'correct':int(margin[i]>0),'top1':int(top1[i])})
                pd.DataFrame(values).drop(columns='Index').to_csv(dest/'measurements.csv',index=False)
                pd.DataFrame(per).to_csv(dest/'per_prompt.csv',index=False)
                print(f'Matched {j}/18 {key}; output_cache_hits={engine.cache.hits}; actual_forwards={meter.counts["forward_batches"]}',flush=True)
                meter.status(stage_group='matched',completed=j,total=18,ranking=key,exact_cache_hits=engine.cache.hits)
        with torch.no_grad():
            after = weight_hash(model)
        assert before == after and all(x.grad is None for x in model.parameters())
        assert all(not x['grad_enabled'] and x['success'] for x in meter.events)
        write_json(out/'model_weight_audit.json',{'passed':True,'before_sha256':before,'after_sha256':after,
                   'stored_model_gradients':False,'backward_calls':0,'optimizer_updates':0})
        write_json(out/'reference_scores.json',engine.refs)
        success = True
    except BaseException as exc:
        import traceback
        write_json(out/'inference_failure.json',{'error':str(exc),'traceback':traceback.format_exc(),
                   'OOM':isinstance(exc,torch.cuda.OutOfMemoryError),'settings_changed':False})
        raise
    finally:
        meter.stop.set();meter.worker.join();meter.save();meter.stream.close()
        write_json(out/'inference_process_summary.json',{'success':success,'projection_stop':stopped,
                   'seconds':time.perf_counter()-start,'counts':meter.counts,
                   'unique_prompts':p['cohort']['size'],'output_cache_hits':engine.cache.hits if engine else 0,
                   'unique_exact_masks':engine.cache.misses if engine else 0,
                   'reference_scalar_hits':engine.reference_hits if engine else 0,
                   'mutable_activation_cache':False})
    write_json(out/'inference_complete.json',{'passed':success and not stopped,'projection_stop':stopped,
               'new_training':False,'Gate5D_decision_modified':False})
