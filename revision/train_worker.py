"""One process per run avoids retained Windows/CUDA resources between models."""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONUTF8','1')
if '--job' in sys.argv:
    _job_path=Path(sys.argv[sys.argv.index('--job')+1])
    _backend=json.loads(_job_path.read_text()).get('spec',{}).get('backend_path')
    if _backend:sys.path.insert(0,_backend)
import torch
from ultralytics import YOLO,RTDETR
import ultralytics
from ultralytics.utils.torch_utils import init_seeds

def create_model(spec, regime, out):
    cls=RTDETR if spec['family']=='RTDETR' else YOLO
    if regime=='pretrained' and 'transfer_map' not in spec:
        return cls(spec['weights'])
    model=cls(spec['architecture'])
    if regime=='pretrained':
        source=YOLO(spec['weights']).model.state_dict();target=model.model.state_dict();matched={};mapping=spec['transfer_map']
        for key,tensor in target.items():
            m=re.match(r'model\.(\d+)\.(.*)',key)
            if m and m.group(1) in mapping:
                source_key=f"model.{mapping[m.group(1)]}.{m.group(2)}"
                if source_key in source and source[source_key].shape==tensor.shape:matched[key]=source[source_key]
        if not matched:raise RuntimeError('No semantically matched COCO tensors')
        model.model.load_state_dict(matched,strict=False)
        # Model.train reconstructs its trainer model and only retains weights when
        # ckpt is truthy. Without this, hand-transferred weights are silently lost.
        model.ckpt = {'semantic_transfer': True}
        (out/'transfer.json').write_text(json.dumps({'checkpoint':spec['weights'],'matched_keys':list(matched),'matched_tensors':len(matched),'total_tensors':len(target),'matched_elements':sum(t.numel() for t in matched.values()),'total_elements':sum(t.numel() for t in target.values()),'policy':'semantic layer map + exact shape; Detect randomly initialized'},indent=2),encoding='utf-8')
    return model

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--qa-stop-after-epoch',type=int);a=p.parse_args();job=json.loads(Path(a.job).read_text());out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(8,os.cpu_count() or 4))
    cfg=job['training']; torch.cuda.reset_peak_memory_stats()
    init_seeds(cfg['seed'],deterministic=cfg['deterministic'])
    last=out/'weights/last.pt'
    if last.exists():
        checkpoint=torch.load(last,map_location='cpu',weights_only=False)
        if checkpoint.get('epoch',-1)<0:raise RuntimeError('Completed/stripped checkpoint without success marker; inspect run before restarting')
        model=(RTDETR if job['spec']['family']=='RTDETR' else YOLO)(str(last))
        result=model.train(resume=True,device=cfg['device'],workers=cfg['workers'])
    else:
        model=create_model(job['spec'],job['regime'],out)
        if a.qa_stop_after_epoch:
            def qa_interrupt(trainer):
                if trainer.epoch+1==a.qa_stop_after_epoch:raise InterruptedError('Intentional QA interruption after checkpoint save')
            model.add_callback('on_model_save',qa_interrupt)
        result=model.train(**cfg,project=str(out.parent),name=out.name,exist_ok=True,pretrained=False,save=True,save_period=10,plots=False,verbose=True)
    trainer=model.trainer
    details={'optimizer_class':type(trainer.optimizer).__name__,'optimizer_defaults':{k:v for k,v in trainer.optimizer.defaults.items() if isinstance(v,(int,float,str,bool,tuple))},'optimizer_groups':[{'lr':g['lr'],'initial_lr':g.get('initial_lr'),'weight_decay':g['weight_decay'],'betas':g.get('betas'),'parameters':sum(p.numel() for p in g['params'])} for g in trainer.optimizer.param_groups],'best_epoch':int(getattr(trainer.stopper,'best_epoch',-1)),'stop_epoch':int(trainer.epoch+1),'cuda_training_peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,'cuda_training_peak_reserved_mb':torch.cuda.max_memory_reserved()/1024**2,'checkpoint':str(out/'weights/best.pt'),'regime':job['regime']}
    (out/'actual_training.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    details['runtime']={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'ultralytics':ultralytics.__version__,'backend_path':job['spec'].get('backend_path'),'attention_backend':job['spec'].get('attention_backend'),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32}
    (out/'actual_training.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    if details['stop_epoch']!=cfg['epochs']:raise RuntimeError(f"Unexpected stop at {details['stop_epoch']}")
    if not (out/'weights/best.pt').exists():raise RuntimeError('Missing best checkpoint')
    (out/'TRAINING_COMPLETE.json').write_text(json.dumps({'job':job,'completed_at':time.time()},indent=2),encoding='utf-8')

if __name__=='__main__':main()
