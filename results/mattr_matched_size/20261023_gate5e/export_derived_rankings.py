from pathlib import Path
import json,gzip,hashlib,torch,pandas as pd,numpy as np
r=Path.cwd();o=r/'results/mattr_matched_size/20261023_gate5e';p=json.loads((o/'gate5e_protocol.json').read_text());items=[]
for rec in p['catalog']:
 if rec['kind']=='original':continue
 scores=torch.load(r/rec['scores'],map_location='cpu',weights_only=False).numpy()
 frame=pd.DataFrame({'canonical_edge_index':np.arange(len(scores)),'signed_score':scores})
 assert np.unique(scores).size==32491
 data=frame.to_csv(index=False).encode();name=rec['key']+'_scores.csv.gz';target=o/'derived_rankings'/name;target.write_bytes(gzip.compress(data,mtime=0))
 items.append({'ranking':rec['key'],'kind':rec['kind'],'csv_gzip':str(target.relative_to(o)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'frozen_tensor_sha256':rec['scores_sha256'],'count':len(scores),'ordering':'descending signed_score; unique negative ordinal values'})
(o/'derived_export_manifest.json').write_text(json.dumps({'scope':'Lossless CSV exports of already frozen derived scores; no transformations or selections changed','rankings':items},indent=2)+'\n')
print('Exported18 full deterministic rankings as small compressed CSVs')
