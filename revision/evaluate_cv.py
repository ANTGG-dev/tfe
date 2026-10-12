"""Evaluate existing CV checkpoints only; summarize seeds within each fold."""
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

METRICS=('AP25','AP50','AP50_95','precision_conf025_iou050','recall_conf025_iou050')

def export(path,rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def summarize(root,jobs):
    rows=[]
    for job in jobs:
        out=Path(job['out'])
        for metric in json.loads((out/'evaluation/metrics.json').read_text(encoding='utf-8')):
            rows.append({'model':job['spec']['id'],'fold':job['fold'],'seed':job['training']['seed'],**metric})
    dest=root/'cv_report';dest.mkdir(exist_ok=True)
    export(dest/'metrics_all_seeds.csv',rows)
    groups=defaultdict(list)
    for row in rows:groups[(row['model'],row['fold'],row['stratum'],row['class'])].append(row)
    folds=[]
    for (model,fold,stratum,cls),values in sorted(groups.items()):
        row={'model':model,'fold':fold,'stratum':stratum,'class':cls,'seeds':len(values)}
        for metric in METRICS:
            valid=[v[metric] for v in values if v[metric] is not None]
            row.update({metric+'_n':len(valid),metric+'_mean':statistics.mean(valid) if valid else None,
                        metric+'_std_seeds':statistics.stdev(valid) if len(valid)>1 else None})
        folds.append(row)
    export(dest/'summary_by_fold.csv',folds)
    grouped=defaultdict(list)
    for row in folds:grouped[(row['model'],row['stratum'],row['class'])].append(row)
    summary=[]
    for (model,stratum,cls),values in sorted(grouped.items()):
        row={'model':model,'stratum':stratum,'class':cls,'folds':len(values)}
        for metric in METRICS:
            valid=[v[metric+'_mean'] for v in values if v[metric+'_mean'] is not None]
            row.update({metric+'_n_folds':len(valid),metric+'_mean_folds':statistics.mean(valid) if valid else None,
                        metric+'_std_folds':statistics.stdev(valid) if len(valid)>1 else None})
        summary.append(row)
    export(dest/'summary_across_folds.csv',summary)
    (dest/'README.md').write_text('CV externa por familias DJI; test original excluido.\n'
        'metrics_all_seeds.csv: resultados de cada corrida.\n'
        'summary_by_fold.csv: media y desviación entre cinco seeds dentro de cada fold.\n'
        'summary_across_folds.csv: media no ponderada y desviación descriptiva de las cinco medias de fold.\n'
        'AP y precision/recall en escala 0–1. Valores ausentes no se convierten en cero.\n'
        'No interpretar los 25 resultados por modelo ni los folds como réplicas independientes.\n'
        'Intervalos bootstrap condicionales por corrida en jobs/.../evaluation/confidence_intervals.json.\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    root=HERE.parent/'V2/mdpi_revision_supreme_v2_640'
    with (nullcontext() if args.check_only else FileLock(str(HERE/'experiment.lock'),timeout=0)):
        frozen=json.loads((root/'frozen_plan.json').read_text(encoding='utf-8'))
        jobs=[j for j in frozen['jobs'] if j['kind']=='cv' and j['dataset']=='local_cv']
        expected={(m,f,s) for m in ('baseline','p2','p2_adown') for f in range(5) for s in range(5)}
        actual={(j['spec']['id'],j['fold'],j['training']['seed']) for j in jobs}
        if len(jobs)!=75 or actual!=expected:raise RuntimeError('Unexpected CV plan')
        repeats=frozen['config']['bootstrap_repeats']
        for job in jobs:
            out=Path(job['out']);marker=out/'TRAINING_COMPLETE.json'
            if not marker.exists() or not (out/'weights/best.pt').exists():
                raise RuntimeError(f'Missing completed training/checkpoint: {out}')
            if json.loads(marker.read_text(encoding='utf-8'))['job']!=job:
                raise RuntimeError(f'Training marker mismatch: {out}')
            if json.loads((out/'job.json').read_text(encoding='utf-8'))!=job:
                raise RuntimeError(f'Job mismatch: {out}')
        original=json.loads((HERE/'prepared/manifest.json').read_text(encoding='utf-8'))
        test_frames={r['frame'] for r in original if r['split']=='test'}
        for fold in range(5):
            path=HERE/'prepared'/f'cv_fold{fold}'/'heldout_manifest.json'
            if frozen['files'].get(str(path))!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise RuntimeError(f'Heldout manifest changed: {path}')
            records=json.loads(path.read_text(encoding='utf-8'))
            if any(r['split']=='test' or r['frame'] in test_frames for r in records):
                raise RuntimeError('Original test family present in CV')
            assignment=json.loads((path.parent/'group_assignment.json').read_text(encoding='utf-8'))
            held=set(assignment['outer_heldout'])
            if held & (set(assignment['inner_train'])|set(assignment['inner_validation'])):
                raise RuntimeError('Outer heldout overlaps development groups')
            for record in records:
                for key,hash_key in (('image','sha256'),('label','label_sha256')):
                    if hashlib.sha256(Path(record[key]).read_bytes()).hexdigest()!=record[hash_key]:
                        raise RuntimeError(f'Changed CV data: {record[key]}')
        print('PASS: 75 completed CV checkpoints, original test excluded, heldout data verified.',flush=True)
        if args.check_only:return
        for i,job in enumerate(jobs,1):
            out=Path(job['out']);marker=out/'evaluation/EVALUATION_COMPLETE.json'
            if marker.exists():
                recorded=json.loads(marker.read_text(encoding='utf-8'))
                if recorded['job_hash']!=canonical_hash(job) or recorded['bootstrap_repeats']!=repeats:
                    raise RuntimeError(f'Evaluation marker mismatch: {out}')
            print(f'Evaluate CV {i}/75: {out.name}',flush=True)
            subprocess.run([sys.executable,'-X','utf8',str(HERE/'eval_worker.py'),
                            '--job',str(out/'job.json'),'--bootstrap',str(repeats)],check=True,cwd=HERE)
        summarize(root,jobs)
        write(root/'cv_report/CV_EVALUATION_COMPLETE.json',{'runs':75,'bootstrap_repeats':repeats,
              'jobs_hash':canonical_hash(jobs),'test_original_used':False})
        print(f'CV complete. Reports: {root / "cv_report"}',flush=True)

if __name__=='__main__':main()
