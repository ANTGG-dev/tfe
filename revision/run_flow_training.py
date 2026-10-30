"""FloW supervised follow-up. Frozen 4 x 5 plan, resumable training, held-out evaluation."""
import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'flow_training'
ROOT = HERE.parent / 'V2/flow_supervised_640'
MODELS = ('p2_adown_tfe_off', 'baseline', 'p2', 'p2_adown')


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    from run_experiment import training_settings
    import torch
    import ultralytics
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    records = json.loads((DATA/'manifest.json').read_text())
    for r in records:
        for key, sha in (('image','sha256'),('label','label_sha256')):
            if digest(Path(r[key])) != r[sha]:
                raise RuntimeError('Dataset modified: '+r[key])
    config = json.loads((HERE/'experiment.json').read_text())
    # Fixed for all 20 runs after sequential, validation-only timing pilots.
    config['workers'] = 4
    # AMP produced an infinite validation classification loss in pilot epoch 1.
    # FP32 pilot had finite losses at both epochs; apply the same choice to all models.
    config['amp'] = False
    catalog = {s['id']:s for s in json.loads((HERE/'model_catalog.json').read_text())}
    plan = []
    for model in MODELS:
        for seed in range(5):
            settings = training_settings(config, DATA/'data.yaml', seed, 640, 'YOLO')
            plan.append({'spec':catalog[model], 'regime':'scratch', 'training':settings, 'out':str(ROOT/'jobs'/f'flow__{model}__scratch__seed{seed}'), 'dataset':'FloW_supervised', 'kind':'main','fold':None})
    sources = [HERE/'train_worker.py', HERE/'run_experiment.py', HERE/'run_flow_training.py', HERE/'analyze_flow_training.py', HERE/'evaluation.py', HERE/'prepare_flow_training.py', DATA/'manifest.json', DATA/'data.yaml', DATA/'audit.json', DATA/'annotation_changes.json']
    sources += [Path(catalog[m]['architecture']) for m in MODELS]
    sources += sorted((HERE.parent/'cuerposAgua/ultralytics_fork/ultralytics').rglob('*.py'))
    snapshot = {'jobs':plan,'files':{str(p):digest(p) for p in sources},'versions':{'torch':torch.__version__,'ultralytics':ultralytics.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},'protocol':{'initialization':'scratch','selection':'validation only','test':'after all 20 trainings','primary_metric':'macro AP50-95','planned_contrasts':[['p2','baseline'],['p2_adown','p2'],['p2_adown_tfe_off','p2_adown']],'native_area_coordinates':'original image dimensions','previous_FloW_exposure':False,'resampling_unit':'image; sequence independence unverified'}}
    target = ROOT/'frozen_plan.json'
    if target.exists() and json.loads(target.read_text()) != snapshot:
        raise RuntimeError('Frozen plan changed. Do not mix configurations in this experiment.')
    write(target,snapshot)
    return plan


def pilot(plan):
    # Same 150-epoch schedule, stop after two saved epochs; separate from final runs.
    job = json.loads(json.dumps(plan[0]))
    out = ROOT/'pilot_fp32'/job['spec']['id']
    job['out'] = str(out)
    job['training']['val'] = True
    jp = out/'job.json'
    if not jp.exists():
        write(jp,job)
        with (out/'console.log').open('w',encoding='utf-8') as log:
            result = subprocess.run([sys.executable,'-X','utf8',str(HERE/'train_worker.py'),'--job',str(jp),'--qa-stop-after-epoch','2'],stdout=log,stderr=subprocess.STDOUT,cwd=HERE)
        message = (out/'console.log').read_text(encoding='utf-8')
        if result.returncode == 0 or 'Intentional QA interruption after checkpoint save' not in message:
            raise RuntimeError('Pilot did not reach its deliberate two-epoch stop; inspect '+str(out/'console.log'))
    rows = list(csv.DictReader((out/'results.csv').open()))
    rows = [{k.strip():v.strip() for k,v in r.items()} for r in rows]
    if len(rows) != 2:
        raise RuntimeError('Pilot incomplete; inspect logs. No full experiment started.')
    import math
    if not all(math.isfinite(float(v)) for row in rows for k,v in row.items() if 'loss' in k):
        raise RuntimeError('Non-finite pilot losses; do not start the final experiment.')
    times = [float(r['time']) for r in rows]
    report = {'model':job['spec']['id'],'epochs_observed':2,'first_epoch_seconds':times[0],'second_epoch_seconds':times[1]-times[0],'conservative_20_runs_150_epochs_hours_at_second_epoch_rate':(times[1]-times[0])*20*150/3600,'note':'Timing pilot only. Includes validation. Other architectures may be faster; not a convergence study. Pilot weights excluded from final experiment. Exit is intentional.'}
    write(ROOT/f'pilot_timing_workers{job["training"]["workers"]}.json',report)
    print(json.dumps(report,indent=2),flush=True)


def evaluate_one(job):
    from ultralytics import YOLO
    from evaluation import predict, evaluate_predictions
    out = Path(job['out'])/'evaluation'
    if (out/'EVALUATION_COMPLETE.json').exists():
        return
    records = [r for r in json.loads((DATA/'manifest.json').read_text()) if r['split']=='test']
    model = YOLO(str(Path(job['out'])/'weights/best.pt'))
    predictions = predict(model,records,640,0)
    evaluate_predictions(records,predictions,out,0,('bottle',))
    write(out/'EVALUATION_COMPLETE.json',{'images':len(records),'selection':'validation only','bootstrap':'performed jointly after all models using images; dependence limitation explicit'})


def summary(plan):
    rows=[]
    for job in plan:
        for result in json.loads((Path(job['out'])/'evaluation/metrics.json').read_text()):
            rows.append({'model':job['spec']['id'],'seed':job['training']['seed'],**result})
    with (ROOT/'results_per_seed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    aggregate=[]
    for m in MODELS:
        for cls in ('macro','bottle'):
            selected=[r for r in rows if r['model']==m and r['class']==cls and r['stratum']=='all']
            row={'model':m,'class':cls,'seeds':len(selected)}
            for metric in ('AP50','AP50_95','precision_conf025_iou050','recall_conf025_iou050'):
                values=[r[metric] for r in selected if r[metric] is not None]
                row[metric+'_mean']=statistics.mean(values) if values else None
                row[metric+'_sd']=statistics.stdev(values) if len(values)>1 else None
            aggregate.append(row)
    write(ROOT/'summary.json',aggregate)
    # Do not mark inference complete without the joint image-bootstrap analysis.
    write(ROOT/'TRAINING_AND_TEST_COMPLETE.json',{'runs':len(plan),'remaining':'paired image-bootstrap uncertainty analysis; manuscript integration'})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('plan','pilot','run'),default='plan')
    parser.add_argument('--eval-job')
    args=parser.parse_args()
    if args.eval_job:
        evaluate_one(json.loads(Path(args.eval_job).read_text()));return
    from filelock import FileLock
    with FileLock(str(HERE/'experiment.lock'),timeout=0):
        plan=freeze()
        print('Frozen 20 FloW runs; 150 epochs; scratch; batch 4; 640 px.',flush=True)
        if args.mode=='pilot':
            pilot(plan)
        elif args.mode=='run':
            from run_experiment import train
            for i,job in enumerate(plan,1):
                print(f'RUN {i}/20: {job["spec"]["id"]} seed {job["training"]["seed"]}',flush=True)
                train(job)
            write(ROOT/'ALL_TRAINING_COMPLETE.json',{'runs':20})
            for job in plan:
                subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--eval-job',str(Path(job['out'])/'job.json')],check=True,cwd=HERE)
            summary(plan)


if __name__=='__main__':
    main()
