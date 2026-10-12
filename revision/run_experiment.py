"""Frozen full-label experiments. All training/selection precedes any TEST inference."""
import argparse
import copy
import hashlib
import json
import os
import platform
import importlib.metadata
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import yaml

HERE=Path(__file__).resolve().parent
os.environ.setdefault('PYTHONUTF8','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def training_settings(config,data,seed,imgsz,family):
    excluded={'name','data','seeds','bootstrap_repeats','resolution_sensitivity','sensitivity_models','grouped_cv_folds','cv_models','public_dataset','public_models'}
    result={k:v for k,v in config.items() if k not in excluded}
    result.update({'data':str(data),'seed':seed,'imgsz':imgsz})
    if family=='RTDETR':
        # Official RT-DETR trainer documents AMP matching NaNs/grid_sample
        # nondeterminism. Run FP32 and explicitly record the exception.
        result.update({'amp':False,'deterministic':False})
    return result

def build_cv(records,count):
    pool=[r for r in records if r['split']!='test'];groups=sorted({r['frame'] for r in pool})
    if count<2:return []
    rng=np.random.default_rng(20260930);rng.shuffle(groups);folds=np.array_split(groups,count);paths=[]
    for i,groups_val in enumerate(folds):
        out=HERE/'prepared'/f'cv_fold{i}';out.mkdir(exist_ok=True);selected=set(groups_val)
        remaining=sorted(set(groups)-selected);inner_rng=np.random.default_rng(20260930+i);inner_rng.shuffle(remaining)
        inner_valid=set(remaining[:max(1,round(.15*len(remaining)))])
        for split in ('train','valid','test'):
            selected_groups=selected if split=='test' else inner_valid if split=='valid' else set(remaining)-inner_valid
            values=[r for r in pool if r['frame'] in selected_groups]
            (out/f'{split}.txt').write_text('\n'.join(str(HERE/'prepared/dataset'/r['split']/'images'/Path(r['image']).name) for r in values)+'\n',encoding='utf-8')
        data={'path':str(out),'train':str(out/'train.txt'),'val':str(out/'valid.txt'),'test':str(out/'test.txt'),'names':{0:'plastic',1:'debris'}}
        (out/'data.yaml').write_text(yaml.safe_dump(data),encoding='utf-8')
        write(out/'heldout_manifest.json',[r for r in pool if r['frame'] in selected]);write(out/'group_assignment.json',{'outer_heldout':sorted(selected),'inner_validation':sorted(inner_valid),'inner_train':sorted(set(remaining)-inner_valid),'original_test_excluded':True});paths.append(out/'data.yaml')
    return paths

def jobs(config,catalog,root,include_public=True):
    records=json.loads((HERE/'prepared/manifest.json').read_text());result=[]
    def add(spec,regime,seed,size,dataset,data,kind='main',fold=None):
        key=f'{dataset}__{kind}__{spec["id"]}__{regime}__img{size}__seed{seed}'+(f'__fold{fold}' if fold is not None else '')
        result.append({'spec':spec,'regime':regime,'training':training_settings(config,data,seed,size,spec['family']),'out':str(root/'jobs'/key),'dataset':dataset,'kind':kind,'fold':fold})
    data=(HERE/config['data']).resolve()
    for spec in catalog:
        for regime in spec['regimes']:
            for seed in config['seeds']:add(spec,regime,seed,config['imgsz'],'local',data)
    for size in config['resolution_sensitivity']:
        for spec in catalog:
            if spec['id'] in config['sensitivity_models']:
                for seed in config['seeds']:add(spec,'scratch',seed,size,'local',data,'resolution')
    for fold,fold_data in enumerate(build_cv(records,config['grouped_cv_folds'])):
        for spec in catalog:
            if spec['id'] in config['cv_models']:
                for seed in config['seeds']:add(spec,'scratch',seed,config['imgsz'],'local_cv',fold_data,'cv',fold)
    if include_public:
        public=HERE/config['public_dataset']
        if not public.exists():raise FileNotFoundError(f'Prepare public benchmark before freezing plan: {public}')
        for spec in catalog:
            if spec['id'] in config['public_models']:
                for seed in config['seeds']:add(spec,'pretrained',seed,config['imgsz'],'UAVVaste',public)
    # Keep every run of the non-YOLO11 version baselines at the end,
    # including the public benchmark. Start with the heaviest measured TFE
    # to exercise real training memory before spending time on other models.
    yolo_versions={'yolov8n','yolov10n','yolo12n','yolov13n','yolo26n'}
    heaviest='legacy_yolo11n-p2-repc3-tfe-tiny'
    return sorted(result,key=lambda job:0 if job['spec']['id']==heaviest else 2 if job['spec']['id'] in yolo_versions else 1)

def train(job):
    out=Path(job['out']);out.mkdir(parents=True,exist_ok=True);path=out/'job.json';marker=out/'TRAINING_COMPLETE.json'
    if path.exists() and json.loads(path.read_text())!=job:raise RuntimeError(f'Configuration changed for {out}; use a new experiment name')
    write(path,job)
    if marker.exists():
        if json.loads(marker.read_text())['job']!=job:raise RuntimeError('Completion marker config mismatch')
        print(f'[SKIP completed] {out.name}',flush=True);return
    print(f'[TRAIN] {out.name}',flush=True)
    with (out/'console.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-X','utf8',str(HERE/'train_worker.py'),'--job',str(path)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',cwd=HERE,env={**os.environ,'PYTHONUTF8':'1'})
        try:
            for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if code or not marker.exists():raise RuntimeError(f'Training failed ({code}): {out}; see console.log. No automatic batch/optimizer changes.')

def evaluate_job(job,repeats):
    from ultralytics import YOLO,RTDETR
    from evaluation import predict,evaluate_predictions
    from profile_runtime import profile
    import torch,gc
    out=Path(job['out']);evaluation=out/'evaluation'
    if (evaluation/'EVALUATION_COMPLETE.json').exists():return
    cls=RTDETR if job['spec']['family']=='RTDETR' else YOLO
    model=cls(str(out/'weights/best.pt'))
    if job['dataset']=='local':records=[r for r in json.loads((HERE/'prepared/manifest.json').read_text()) if r['split']=='test'];names=('plastic','debris')
    elif job['dataset']=='local_cv':records=json.loads((HERE/'prepared'/f'cv_fold{job["fold"]}'/'heldout_manifest.json').read_text());names=('plastic','debris')
    else:records=[r for r in json.loads((HERE/'public/prepared/manifest.json').read_text()) if r['split']=='test'];names=('rubbish',)
    predictions=predict(model,records,job['training']['imgsz'],job['training']['device'])
    evaluate_predictions(records,predictions,evaluation,repeats,names)
    if job['kind']!='cv':
        write(out/'profiling.json',profile(model,job['training']['imgsz']))
    write(evaluation/'EVALUATION_COMPLETE.json',{'job_hash':canonical_hash(job),'images':len(records),'bootstrap_repeats':repeats,'selection_policy':'best.pt selected exclusively using validation; all jobs trained before held-out evaluation'})
    del model;gc.collect();torch.cuda.empty_cache()

def execute_main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(HERE/'experiment.json'));p.add_argument('--plan-only',action='store_true');p.add_argument('--local-only',action='store_true');a=p.parse_args()
    config=json.loads(Path(a.config).read_text());catalog=json.loads((HERE/'model_catalog.json').read_text());root=HERE.parent/'V2'/config['name'];root.mkdir(parents=True,exist_ok=True)
    import torch,ultralytics
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refusing CPU experiment')
    if not (HERE/'prepared/manifest.json').exists():raise RuntimeError('Run audit_dataset.py first')
    records=json.loads((HERE/'prepared/manifest.json').read_text())
    for record in records:
        for key,hash_key in (('image','sha256'),('label','label_sha256')):
            if hashlib.sha256(Path(record[key]).read_bytes()).hexdigest()!=record[hash_key]:raise RuntimeError(f'Prepared data changed: {record[key]}')
    if not a.local_only and not (HERE/config['public_dataset']).exists():
        subprocess.run([sys.executable,'-X','utf8',str(HERE/'prepare_public.py')],check=True)
    if not a.local_only:
        for record in json.loads((HERE/'public/prepared/manifest.json').read_text()):
            for key,hash_key in (('image','sha256'),('label','label_sha256')):
                if hashlib.sha256(Path(record[key]).read_bytes()).hexdigest()!=record[hash_key]:raise RuntimeError(f'Public prepared data changed: {record[key]}')
    plan=jobs(config,catalog,root,not a.local_only)
    # Freeze code, YAML, labels, split manifests, configuration and versions.
    files=list(HERE.glob('*.py'))+list(HERE.glob('models/*.yaml'))+[HERE/'prepared/manifest.json',HERE/'prepared/data.yaml',HERE/'prepared/provenance.json',Path(a.config)]
    files+=list((HERE/'prepared').glob('cv_fold*/*'))
    for spec in catalog:
        path=Path(spec['architecture'])
        if path.exists():files.append(path)
    files+=list((HERE.parent/'cuerposAgua/ultralytics_fork/ultralytics').rglob('*.py'))
    for spec in catalog:
        if spec.get('backend_path'):files+=list(Path(spec['backend_path']).joinpath('ultralytics').rglob('*.py'))
        if spec.get('weights') and Path(spec['weights']).exists():files.append(Path(spec['weights']))
    if not a.local_only:files+=[HERE/'public/prepared/manifest.json',HERE/config['public_dataset']]
    snapshot={'jobs':plan,'files':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files},'versions':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'ultralytics':ultralytics.__version__,'gpu':torch.cuda.get_device_name(0),'packages':dict(sorted((d.metadata['Name'].lower(),d.version) for d in importlib.metadata.distributions()))},'config':config}
    frozen=root/'frozen_plan.json'
    if frozen.exists() and json.loads(frozen.read_text())!=snapshot:raise RuntimeError('Frozen plan changed; use a new experiment name. Never adapt this plan using test results.')
    write(frozen,snapshot)
    write(root/'environment.json',{'platform':platform.platform(),'versions':snapshot['versions'],'nvidia_smi':subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,'packages':subprocess.run([str(HERE/'tools/uv.exe'),'pip','freeze','--python',sys.executable],capture_output=True,text=True).stdout})
    print(f'Frozen {len(plan)} training runs. Output: {root}',flush=True)
    if a.plan_only:return
    try:
        for i,job in enumerate(plan,1):print(f'Run {i}/{len(plan)}',flush=True);train(job)
        write(root/'ALL_TRAINING_COMPLETE.json',{'plan_hash':canonical_hash(snapshot),'runs':len(plan),'timestamp':time.time()})
        for i,job in enumerate(plan,1):
            print(f'Evaluate {i}/{len(plan)} {Path(job["out"]).name}',flush=True)
            subprocess.run([sys.executable,'-X','utf8',str(HERE/'eval_worker.py'),'--job',str(Path(job['out'])/'job.json'),'--bootstrap',str(config['bootstrap_repeats'])],check=True,cwd=HERE)
        from statistics_report import report
        report(root)
        write(root/'EXPERIMENT_COMPLETE.json',{'runs':len(plan),'timestamp':time.time()})
    except BaseException as exc:
        with (root/'failures.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps({'timestamp':time.time(),'type':type(exc).__name__,'error':str(exc)})+'\n')
        raise

def main():
    from filelock import FileLock
    with FileLock(str(HERE/'experiment.lock'),timeout=0):execute_main()

if __name__=='__main__':main()
