"""Final local TEST evaluation; fixed baseline and ablation contrasts, no training."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from filelock import FileLock
from run_experiment import HERE, canonical_hash, write

VERSIONS={'yolov8n','yolov10n','yolo12n','yolo26n','yolov13n'}
ABLATIONS=[('p2_adown','p2'),('p2_adown','adown'),('p2_tfe_off','p2'),
 ('p2_tfe_on','p2_tfe_off'),('p2_adown_tfe_off','p2_adown'),
 ('p2_adown_tfe_on','p2_adown_tfe_off'),('p2_adown_repc3','p2_adown'),
 ('p2_adown_repc3_tfe_off','p2_adown_repc3'),
 ('p2_adown_repc3_tfe_on','p2_adown_repc3_tfe_off')]

def export(path,rows):
    if not rows:raise RuntimeError(f'No rows: {path}')
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def report(root,jobs):
    import numpy as np
    from scipy import stats
    dest=root/'test_report';dest.mkdir(exist_ok=True)
    rows=[];groups=defaultdict(list)
    for job in jobs:
        out=Path(job['out']);regime=job['regime'];model=job['spec']['id']
        metrics=json.loads((out/'evaluation/metrics.json').read_text())
        for item in metrics:
            rows.append({'model':model,'regime':regime,'seed':job['training']['seed'],**item})
        groups[(regime,model)].append((job,metrics))
    export(dest/'test_metrics_all_seeds.csv',rows)
    summary=[];by=defaultdict(list)
    for row in rows:by[(row['model'],row['regime'],row['stratum'],row['class'])].append(row)
    for (model,regime,stratum,cls),values in sorted(by.items()):
        row={'model':model,'regime':regime,'stratum':stratum,'class':cls,'seeds':len(values)}
        for metric in ('AP25','AP50','AP50_95','precision_conf025_iou050','recall_conf025_iou050'):
            v=[r[metric] for r in values if r[metric] is not None]
            row.update({metric+'_n':len(v),metric+'_mean':statistics.mean(v) if v else None,
                        metric+'_std':statistics.stdev(v) if len(v)>1 else None})
        summary.append(row)
    export(dest/'summary_mean_std.csv',summary)
    contrasts=[];welch=[]
    for regime in sorted({r for r,m in groups}):
        models={m for r,m in groups if r==regime}
        pairs=sorted({(m,'baseline') for m in models if m!='baseline'} |
                     {(a,b) for a,b in ABLATIONS if a in models and b in models})
        for model,reference in pairs:
            a={j['training']['seed']:(j,ms) for j,ms in groups[(regime,model)]}
            b={j['training']['seed']:(j,ms) for j,ms in groups[(regime,reference)]}
            seeds=sorted(set(a)&set(b))
            template=a[seeds[0]][1]
            for seed in seeds:
                if [(m['stratum'],m['class']) for m in a[seed][1]] != [(m['stratum'],m['class']) for m in b[seed][1]]:
                    raise RuntimeError('Metric ordering mismatch')
            for metric in ('AP50','AP50_95'):
                for i,m in enumerate(template):
                    x=[a[s][1][i][metric] for s in seeds];y=[b[s][1][i][metric] for s in seeds]
                    valid=[(u,v) for u,v in zip(x,y) if u is not None and v is not None]
                    if len(valid)<2:continue
                    x,y=map(np.asarray,zip(*valid));res=stats.ttest_ind(x,y,equal_var=False)
                    vx=np.var(x,ddof=1)/len(x);vy=np.var(y,ddof=1)/len(y)
                    df=(vx+vy)**2/(vx**2/(len(x)-1)+vy**2/(len(y)-1)) if vx+vy else None
                    welch.append({'regime':regime,'model':model,'reference':reference,'stratum':m['stratum'],
                         'class':m['class'],'metric':metric,'n':len(valid),'mean_difference':float(np.mean(x-y)),
                         'welch_t':float(res.statistic) if np.isfinite(res.statistic) else None,'df':df,
                         'raw_p':float(res.pvalue) if np.isfinite(res.pvalue) else None,
                         'q_BH':None,'family_size':None,'interpretation':'exploratory; shared images and paired seeds limit Welch independence'})
                for unit in ('image','frame'):
                    draws=[]
                    for s in seeds:
                        left=Path(a[s][0]['out'])/'evaluation'/f'bootstrap_{unit}_{metric}.npy'
                        right=Path(b[s][0]['out'])/'evaluation'/f'bootstrap_{unit}_{metric}.npy'
                        draws.append(np.load(left)-np.load(right))
                    stacked=np.stack(draws)
                    # Require all seed contrasts to be defined for a draw.
                    delta=np.mean(stacked,axis=0)
                    for i,m in enumerate(template):
                        valid=delta[:,i];valid=valid[np.isfinite(valid)]
                        values=[a[s][1][i][metric]-b[s][1][i][metric] for s in seeds
                                if a[s][1][i][metric] is not None and b[s][1][i][metric] is not None]
                        contrasts.append({'regime':regime,'model':model,'reference':reference,'stratum':m['stratum'],
                          'class':m['class'],'metric':metric,'seeds':len(seeds),'mean_difference':statistics.mean(values) if values else None,
                          'bootstrap_unit':unit,'ci_low':float(np.quantile(valid,.025)) if len(valid) else None,
                          'ci_high':float(np.quantile(valid,.975)) if len(valid) else None,'valid_draws':len(valid),
                          'interpretation':'pointwise paired 95% percentile CI, conditional on fitted seeds; frame primary; not equivalence'})
        family=[r for r in welch if r['regime']==regime and r['raw_p'] is not None]
        order=sorted(range(len(family)),key=lambda i:family[i]['raw_p']);last=1.
        for rank in range(len(order)-1,-1,-1):
            row=family[order[rank]];last=min(last,row['raw_p']*len(order)/(rank+1))
            row.update({'q_BH':last,'family_size':len(order)})
    export(dest/'paired_bootstrap_differences.csv',contrasts)
    export(dest/'welch_BH_exploratory.csv',welch)
    (dest/'README.md').write_text('All metrics use scale 0–1. Each regime is compared separately.\n'
      'Fixed contrasts: every model vs baseline plus component ablations; not all pairs.\n'
      'Bootstrap CIs are pointwise, conditional on fitted models, not corrected simultaneous intervals.\n'
      'Frame-family bootstrap is primary; image bootstrap is supplementary.\n'
      'Welch/BH is exploratory, family = all available fixed contrasts/classes/strata/AP metrics within regime.\n'
      'CI including zero does not prove equivalence. Flight independence remains unverified.\n'
      'Per-run CIs and profiling remain in jobs/.../evaluation and profiling.json.\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');args=p.parse_args()
    root=HERE.parent/'V2/mdpi_revision_supreme_v2_640'
    with (nullcontext() if args.check_only else FileLock(str(HERE/'experiment.lock'),timeout=0)):
        frozen=json.loads((root/'frozen_plan.json').read_text())
        jobs=[j for j in frozen['jobs'] if j['dataset']=='local' and j['kind']=='main'
              and j['spec']['id']!='yolov13n'
              and (j['spec']['id'] not in VERSIONS or j['regime']=='scratch')]
        if len(jobs)!=150:raise RuntimeError(f'Unexpected local plan size: {len(jobs)}')
        repeats=frozen['config']['bootstrap_repeats'];missing=[]
        for j in jobs:
            out=Path(j['out']);marker=out/'TRAINING_COMPLETE.json'
            if not marker.exists() or not (out/'weights/best.pt').exists():missing.append(out.name);continue
            if json.loads(marker.read_text())['job']!=j or json.loads((out/'job.json').read_text())!=j:
                raise RuntimeError(f'Job mismatch: {out}')
        manifest_path=HERE/'prepared/manifest.json'
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=frozen['files'][str(manifest_path)]:
            raise RuntimeError('Dataset manifest changed')
        records=[r for r in json.loads(manifest_path.read_text()) if r['split']=='test']
        for r in records:
            for key,h in (('image','sha256'),('label','label_sha256')):
                if hashlib.sha256(Path(r[key]).read_bytes()).hexdigest()!=r[h]:raise RuntimeError('Test data changed')
        print(f'Plan: {len(jobs)} local runs, {len(records)} test images; {len(missing)} trainings pending.',flush=True)
        if args.check_only:return
        if missing:raise RuntimeError('Finish training first: '+', '.join(missing))
        write(root/'test_evaluation_plan.json',{'jobs':jobs,'bootstrap_repeats':repeats,'ablation_pairs':ABLATIONS,
               'policy':'local main runs through YOLO26; YOLO13 excluded by user; other YOLO scratch only; no UAVVaste or CV'})
        for i,j in enumerate(jobs,1):
            out=Path(j['out']);marker=out/'evaluation/EVALUATION_COMPLETE.json'
            if marker.exists():
                m=json.loads(marker.read_text())
                if m['job_hash']!=canonical_hash(j) or m['bootstrap_repeats']!=repeats:raise RuntimeError('Evaluation marker mismatch')
            print(f'Test {i}/{len(jobs)} {out.name}',flush=True)
            subprocess.run([sys.executable,'-X','utf8',str(HERE/'eval_worker.py'),'--job',str(out/'job.json'),
                            '--bootstrap',str(repeats)],check=True,cwd=HERE)
        report(root,jobs)
        write(root/'test_report/TEST_EVALUATION_COMPLETE.json',{'runs':len(jobs),'jobs_hash':canonical_hash(jobs)})

if __name__=='__main__':main()
