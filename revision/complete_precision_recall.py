"""Fixed-threshold P/R uncertainty, conditional on the existing fitted seeds.

Reuses COCO matching at confidence .25 / IoU .50. Resampling counts is exact
because detection matching is independent within images. No inference/training.
"""
from pathlib import Path
import collections, csv, json
import numpy as np
from evaluation import evaluate, RANGES, STRATA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'revision/manuscript_revision/supplement'

def counts_from_evaluator(e):
    ids = sorted(e.params.imgIds)
    cats = list(e.cocoGt.dataset['categories'])
    counts = np.zeros((len(ids), len(STRATA), len(cats)+1, 3), dtype=np.int64)
    ii = {x:i for i,x in enumerate(ids)}
    kk = {c['id']:i+1 for i,c in enumerate(cats)}
    for item in e.evalImgs:
        if item is None: continue
        a = RANGES.index(item['aRng']); k = kk[item['category_id']]
        ignore = item['dtIgnore'][1]; match = item['dtMatches'][1] > 0
        v = [np.sum(match & ~ignore), np.sum(~match & ~ignore),
             np.sum(~np.asarray(item['gtIgnore'], bool))]
        counts[ii[item['image_id']], a, k] = v
    counts[:,:,0] = counts[:,:,1:].sum(axis=2)
    return counts.reshape(len(ids), -1, 3), cats

def ratios(c):
    tp, fp, gt = (c[...,i].astype(float) for i in range(3))
    p = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=tp+fp>0)
    r = np.divide(tp, gt, out=np.full_like(tp, np.nan), where=gt>0)
    return np.stack([p,r], axis=-1)

def weights_for(records, unit, repeats=1000):
    groups = collections.defaultdict(list)
    for i,r in enumerate(records): groups[r['frame'] if unit=='frame' else str(i)].append(i)
    keys = sorted(groups); rng = np.random.default_rng(20260930)
    weights = np.zeros((repeats,len(records)), np.int64)
    for b in range(repeats):
        for key in rng.choice(keys,len(keys),replace=True): weights[b,groups[key]] += 1
    return weights, len(keys)

def check_resampling():
    # Includes repeated images, a negative image with a false positive, and
    # absent GT. Checks ratios of aggregated counts, not averages of ratios.
    c=np.array([[[2,1,3],[0,0,0]],[[0,2,0],[0,1,0]],[[1,0,2],[0,0,0]]])
    indices=[2,0,0,1]; w=np.bincount(indices,minlength=3)
    assert np.array_equal(np.einsum('i,ijk->jk',w,c), c[indices].sum(0))
    np.testing.assert_allclose(ratios(c[indices].sum(0))[0],[5/9,5/8])
    assert np.isnan(ratios(c.sum(0))[1,1])

def main():
    check_resampling()
    records=[r for r in json.loads((ROOT/'revision/prepared/manifest.json').read_text()) if r['split']=='test']
    plan=json.loads((ROOT/'V2/mdpi_revision_supreme_v2_640/test_evaluation_plan.json').read_text())['jobs']
    pairs=json.loads((ROOT/'V2/batch8_sensitivity/plan.json').read_text())['pairs']
    jobs=[('main',j) for j in plan]+[('batch8',p['batch8_job']) for p in pairs]
    weights={u:weights_for(records,u) for u in ('frame','image')}
    grouped=collections.defaultdict(list); checks=[]
    cache=ROOT/'revision/fixed_pr_evidence';cache.mkdir(exist_ok=True)
    for n,(scope,j) in enumerate(jobs,1):
        folder=Path(j['out'])/'evaluation'
        gt=json.loads((folder/'ground_truth.json').read_text())
        assert [im['file_name'] for im in gt['images']]==[Path(r['image']).name for r in records]
        e=evaluate(gt,json.loads((folder/'predictions.json').read_text()),.25)
        c,cats=counts_from_evaluator(e); full=c.sum(0); point=ratios(full)
        saved=json.loads((folder/'metrics.json').read_text())
        for i,row in enumerate(saved):
            assert list(full[i])==[row[k] for k in ('TP','FP','GT')],(j['out'],i)
            for k,key in enumerate(('precision_conf025_iou050','recall_conf025_iou050')):
                expected=row[key]
                assert (expected is None and np.isnan(point[i,k])) or np.isclose(point[i,k],expected,rtol=0,atol=1e-12)
        key=(scope,j['spec']['id'],j['regime'])
        draws={u:ratios(np.einsum('bi,ijk->bjk',w,c)) for u,(w,_) in weights.items()}
        np.savez_compressed(cache/(Path(j['out']).name+'.npz'),counts=c,point=point,**draws)
        grouped[key].append((point,draws))
        checks.append({'scope':scope,'job':Path(j['out']).name,'rows_verified':len(saved),'counts_and_metrics_match':True})
        if n%10==0:print(f'P/R: {n}/{len(jobs)} jobs verified',flush=True)
    output=[]
    for (scope,model,regime),runs in grouped.items():
        assert len(runs)==5
        points=np.stack([r[0] for r in runs]); mean=points.mean(0); sd=points.std(0,ddof=1)
        for unit,(_,ngroups) in weights.items():
            draws=np.stack([r[1][unit] for r in runs]).mean(0)
            for a,s in enumerate(STRATA):
                for k,cl in enumerate(['macro']+[c['name'] for c in cats]):
                    for m,metric in enumerate(['precision_conf025_iou050','recall_conf025_iou050']):
                        i=a*3+k; v=draws[:,i,m];v=v[np.isfinite(v)]
                        lo,hi=np.quantile(v,[.025,.975]) if len(v) else [np.nan,np.nan]
                        output.append(dict(scope=scope,model=model,regime=regime,stratum=s,**{'class':cl},metric=metric,bootstrap_unit=unit,bootstrap_groups=ngroups,mean=mean[i,m],seed_sd=sd[i,m],ci_low=lo,ci_high=hi,valid_draws=len(v),repeats=1000,fitted_seeds=5))
    OUT.mkdir(exist_ok=True)
    with (OUT/'S13_fixed_precision_recall_intervals.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=output[0]);w.writeheader();w.writerows(output)
    (cache/'verification.json').write_text(json.dumps({'checks':checks,'rows':len(output),'resampling_test':'passed','confidence':.25,'iou':.5},indent=2))
    print(f'COMPLETE: {len(jobs)} jobs, {len(output)} CI rows',flush=True)

if __name__=='__main__': main()
