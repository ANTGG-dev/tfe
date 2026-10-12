"""Descriptive seed variability, Welch/BH, paired dataset-bootstrap differences."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def bh(values):
    values=np.asarray(values);order=np.argsort(values);ranked=values[order]*len(values)/np.arange(1,len(values)+1)
    result=np.empty(len(values));result[order]=np.minimum(1,np.minimum.accumulate(ranked[::-1])[::-1]);return result

def report(root):
    root=Path(root);rows=[]
    for path in sorted(root.glob('jobs/*/evaluation/metrics.json')):
        job=json.loads((path.parents[1]/'job.json').read_text())
        for row in json.loads(path.read_text()):rows.append({**row,'model':job['spec']['id'],'regime':job['regime'],'seed':job['training']['seed'],'imgsz':job['training']['imgsz'],'dataset':job['dataset'],'run':path.parents[1].name,'evaluation_dir':str(path.parent)})
    if not rows:raise ValueError('No final test metrics')
    df=pd.DataFrame(rows);df.to_csv(root/'test_metrics_all_seeds.csv',index=False)
    keys=['dataset','regime','imgsz','model','stratum','class']
    df.groupby(keys,dropna=False)[['AP25','AP50','AP50_95','precision_conf025_iou050','recall_conf025_iou050']].agg(['count','mean','std']).to_csv(root/'summary_mean_std.csv')
    comparisons=[]
    families=['dataset','regime','imgsz']
    for family,subset in df.groupby(families):
        if family[0]=='local_cv':continue  # overlapping CV training folds are not independent Welch observations
        baseline=subset[subset.model.eq('baseline')]
        pending=[]
        for (model,stratum,cls),group in subset[~subset.model.eq('baseline')].groupby(['model','stratum','class']):
            b=baseline[baseline.stratum.eq(stratum)&baseline['class'].eq(cls)]
            for metric in ('AP50','AP50_95'):
                x=group[metric].dropna().to_numpy();y=b[metric].dropna().to_numpy()
                if len(x)<2 or len(y)<2:continue
                vx=np.var(x,ddof=1)/len(x);vy=np.var(y,ddof=1)/len(y)
                dfree=(vx+vy)**2/(vx**2/(len(x)-1)+vy**2/(len(y)-1)) if vx+vy else None
                res=stats.ttest_ind(x,y,equal_var=False)
                pending.append({'dataset':family[0],'regime':family[1],'imgsz':family[2],'model':model,'reference':'baseline','stratum':stratum,'class':cls,'metric':metric,'n_model':len(x),'n_reference':len(y),'mean_difference':float(x.mean()-y.mean()),'welch_t':float(res.statistic) if np.isfinite(res.statistic) else None,'degrees_freedom':dfree,'raw_p':float(res.pvalue) if np.isfinite(res.pvalue) else None,'interpretation':'exploratory optimization-seed comparison; shared dataset and paired seeds limit Welch independence'})
        valid=[r for r in pending if r['raw_p'] is not None]
        for r,q in zip(valid,bh([r['raw_p'] for r in valid])):r.update({'adjusted_q_BH':float(q),'multiple_comparison_family_size':len(valid),'family_definition':'all baseline contrasts, both AP metrics, every class and stratum, within dataset/regime/resolution'})
        comparisons.extend(pending)
    pd.DataFrame(comparisons).to_csv(root/'welch_BH_exploratory.csv',index=False)
    paired=[]
    for family,subset in df.groupby(families):
        if family[0]=='local_cv':continue
        runs=subset.drop_duplicates('run');base=runs[runs.model.eq('baseline')]
        for model,group in runs[~runs.model.eq('baseline')].groupby('model'):
            common=sorted(set(group.seed)&set(base.seed))
            if not common:continue
            for unit in ('image','frame'):
                differences=[]
                for seed in common:
                    a=Path(group[group.seed.eq(seed)].iloc[0].evaluation_dir)/f'bootstrap_{unit}.npy';b=Path(base[base.seed.eq(seed)].iloc[0].evaluation_dir)/f'bootstrap_{unit}.npy'
                    if a.exists() and b.exists():differences.append(np.load(a)-np.load(b))
                if not differences:continue
                # Resampling draws are identical across models/seeds, yielding paired contrasts.
                with np.errstate(invalid='ignore'):delta=np.nanmean(np.stack(differences),axis=0)
                metrics=json.loads((Path(group.iloc[0].evaluation_dir)/'metrics.json').read_text())
                for i,m in enumerate(metrics):
                    valid=delta[:,i];valid=valid[np.isfinite(valid)]
                    paired.append({'dataset':family[0],'regime':family[1],'imgsz':family[2],'model':model,'reference':'baseline','stratum':m['stratum'],'class':m['class'],'metric':'AP50_95','ci_low':float(np.quantile(valid,.025)) if len(valid) else None,'ci_high':float(np.quantile(valid,.975)) if len(valid) else None,'bootstrap_unit':unit,'seeds_averaged':len(differences),'valid_draws':len(valid),'interpretation':'paired percentile CI for mean-seed difference, conditional on fitted models; frame clustering primary'})
    pd.DataFrame(paired).to_csv(root/'paired_bootstrap_differences.csv',index=False)
