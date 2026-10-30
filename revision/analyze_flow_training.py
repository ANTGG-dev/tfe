"""Paired image bootstrap, conditional on the 20 fitted models."""
import hashlib
import json
from pathlib import Path

import numpy as np
from evaluation import bootstrap, evaluate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent/'V2/flow_supervised_640'
CONTRASTS = [('p2','baseline'),('p2_adown','p2'),('p2_adown_tfe_off','p2_adown')]


def main():
    plan = json.loads((ROOT/'frozen_plan.json').read_text())['jobs']
    records = [r for r in json.loads((HERE/'flow_training/manifest.json').read_text()) if r['split']=='test']
    arrays = {}
    for job in plan:
        folder = Path(job['out'])/'evaluation'
        assert (folder/'EVALUATION_COMPLETE.json').exists(), 'All test evaluations must finish first'
        cache = folder/'paired_image_draws.npz'
        source_hash = hashlib.sha256((folder/'predictions.json').read_bytes()+(folder/'ground_truth.json').read_bytes()).hexdigest()
        meta = folder/'paired_image_draws_metadata.json'
        expected = {'source_hash':source_hash,'repeats':1000,'seed':20261028,'unit':'image'}
        if cache.exists():
            assert json.loads(meta.read_text()) == expected
        else:
            print('Bootstrap:',job['spec']['id'],job['training']['seed'],flush=True)
            evaluator = evaluate(json.loads((folder/'ground_truth.json').read_text()),json.loads((folder/'predictions.json').read_text()))
            _, draws = bootstrap(evaluator,records,repeats=1000,seed=20261028,unit='image')
            np.savez(cache,**draws)
            meta.write_text(json.dumps(expected,indent=2))
        with np.load(cache) as values:
            # metrics ordering: all/macro and all/bottle first.
            arrays[(job['spec']['id'],job['training']['seed'])] = values['AP50_95'][:,:2].copy()
    results=[]
    summaries = {(r['model'],r['class']):r for r in json.loads((ROOT/'summary.json').read_text())}
    for model,reference in CONTRASTS:
        delta=np.mean(np.stack([arrays[(model,s)]-arrays[(reference,s)] for s in range(5)]),axis=0)*100
        result={'model':model,'reference':reference,'metric':'AP50-95','units':'percentage points','bootstrap_repeats':1000,'images':len(records),'conditional_on_fitted_models':True}
        for i,name in enumerate(('macro','bottle')):
            values=delta[:,i];values=values[np.isfinite(values)]
            point = 100*(summaries[(model,name)]['AP50_95_mean']-summaries[(reference,name)]['AP50_95_mean'])
            result[name]={'mean_difference':point,'pointwise_95_ci':np.quantile(values,[.025,.975]).tolist(),'valid_draws':len(values)}
            if i==0:
                result[name]['bonferroni_3_contrasts_98_333_ci']=np.quantile(values,[.05/6,1-.05/6]).tolist()
        results.append(result)
    (ROOT/'paired_image_contrasts.json').write_text(json.dumps(results,indent=2))
    (ROOT/'EXPERIMENT_COMPLETE.json').write_text(json.dumps({'runs':20,'analysis':'paired image bootstrap; 3 planned macro contrasts; pointwise class intervals; training-seed SD reported separately','limitations':'Conditional intervals do not include new-training uncertainty. Image resampling assumes independence not established for related scenes; CIs may be optimistic.'},indent=2))
    print('Training, test evaluation and image-bootstrap analysis complete.',flush=True)


if __name__=='__main__':
    main()
