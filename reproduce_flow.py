"""Portable FloW entry point; published evidence is never overwritten."""
import argparse, json, shutil, sys, csv, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EVIDENCE=ROOT/'evidence/flow'
sys.path.insert(0,str(ROOT/'revision'))
def expand(value):
    if isinstance(value,dict):return {k:expand(v) for k,v in value.items()}
    if isinstance(value,list):return [expand(v) for v in value]
    return value.replace('${ROOT}',ROOT.as_posix()).replace('\\','/') if isinstance(value,str) and '${ROOT}' in value else value
def verify_results():
    import numpy as np
    summary=json.loads((EVIDENCE/'summary.json').read_text())
    plan=json.loads((EVIDENCE/'frozen_plan.json').read_text())['jobs']
    arrays={};metrics={}
    for j in plan:
        model=j['spec']['id'];seed=j['training']['seed']
        p=EVIDENCE/'jobs'/Path(j['out'].replace('\\','/')).name
        with (p/'results.csv').open() as f:rows=list(csv.DictReader(f))
        assert int(float({k.strip():v for k,v in rows[-1].items()}['epoch']))==150
        assert (p/'weights/best.pt').is_file()
        rows=json.loads((p/'evaluation/metrics.json').read_text())
        metrics[model,seed]=next(r for r in rows if r['stratum']=='all' and r['class']=='macro')
        with np.load(p/'evaluation/paired_image_draws.npz') as a:arrays[model,seed]=a['AP50_95'][:,:2].copy()
    for r in summary:
        for key in ['AP50','AP50_95','precision_conf025_iou050','recall_conf025_iou050']:
            v=[metrics[r['model'],i][key] for i in range(5)]
            assert abs(statistics.mean(v)-r[key+'_mean'])<1e-12
            assert abs(statistics.stdev(v)-r[key+'_sd'])<1e-12
    for r in json.loads((EVIDENCE/'paired_image_contrasts.json').read_text()):
        delta=np.mean(np.stack([arrays[r['model'],s]-arrays[r['reference'],s] for s in range(5)]),axis=0)*100
        for c,name in enumerate(['macro','bottle']):
            v=delta[:,c];v=v[np.isfinite(v)]
            assert len(v)==r[name]['valid_draws']
            assert np.allclose(np.quantile(v,[.025,.975]),r[name]['pointwise_95_ci'],atol=1e-12)
            if c==0:assert np.allclose(np.quantile(v,[.05/6,1-.05/6]),r[name]['bonferroni_3_contrasts_98_333_ci'],atol=1e-12)
    print('Verified all 20 final epochs, best checkpoints, reported means/SD and paired-bootstrap intervals. No training started.')
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--verify-results',action='store_true')
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--run',action='store_true')
    p.add_argument('--evaluate-best',action='store_true')
    a=p.parse_args()
    if a.verify_results:verify_results()
    if a.run and a.evaluate_best:p.error('Choose --run or --evaluate-best, not both.')
    if a.prepare:
        import prepare_flow_training
        prepare_flow_training.main()
    if not (a.prepare or a.run or a.evaluate_best):return
    manifest=ROOT/'revision/flow_training/manifest.json'
    actual=json.loads(manifest.read_text());expected=json.loads((EVIDENCE/'data/manifest.json').read_text())
    keys=['uid','split','sha256','label_sha256','width','height','boxes']
    assert [{k:r[k] for k in keys} for r in actual]==[{k:r[k] for k in keys} for r in expected], 'Prepared data differs from frozen evidence.'
    print('Verified original 960/240/800 split and all image/label hashes.')
    if not (a.run or a.evaluate_best):return
    import run_flow_training as runner
    import analyze_flow_training as analyzer
    from run_experiment import train
    from filelock import FileLock
    mode='retrained' if a.run else 'published_best'
    out=ROOT/'runs/flow'/mode
    runner.ROOT=analyzer.ROOT=out
    plan=expand(json.loads((EVIDENCE/'frozen_plan.json').read_text()))
    for j in plan['jobs']:
        name=Path(j['out'].replace('\\','/')).name
        j['out']=str(out/'jobs'/name)
        if a.evaluate_best:
            dest=Path(j['out'])/'weights/best.pt';dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(EVIDENCE/'jobs'/name/'weights/best.pt',dest)
    out.mkdir(parents=True,exist_ok=True)
    target=out/'frozen_plan.json'
    if target.exists():assert json.loads(target.read_text())==plan,'Existing reproduction plan differs.'
    else:target.write_text(json.dumps(plan,indent=2))
    with FileLock(str(ROOT/'revision/experiment.lock'),timeout=0):
        if a.run:
            for j in plan['jobs']:train(j)
        for j in plan['jobs']:runner.evaluate_one(j)
        runner.summary(plan['jobs'])
        analyzer.main()
if __name__=='__main__':main()
