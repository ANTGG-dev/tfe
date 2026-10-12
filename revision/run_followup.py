"""Sequential RiSID inference, then batch-8 training and held-out evaluation."""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
os.environ.setdefault('PYTHONUTF8','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj),encoding='utf-8');tmp.replace(p)

def external(index,smoke=False):
    from ultralytics import YOLO
    from evaluation import evaluate,metrics
    plan=read(HERE/'risid/plan.json');job=plan['jobs'][index]
    out=ROOT/'V2/risid_external'/('smoke' if smoke else 'jobs')/Path(job['out']).name
    out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists(): return
    gt=read(HERE/'risid/ground_truth_plastic.json')
    images=gt['images'][:2] if smoke else gt['images']
    if smoke:
        gt['images']=images;ids={i['id'] for i in images}
        gt['annotations']=[a for a in gt['annotations'] if a['image_id'] in ids]
    model=YOLO(str(Path(job['out'])/'weights/best.pt'))
    assert model.names[0]=='plastic',model.names
    predictions=[]
    for start in range(0,len(images),100):
        chunk=out/'chunks'/f'{start:06d}.json'
        if chunk.exists(): predictions.extend(read(chunk));continue
        current=[]
        for image in images[start:start+100]:
            result=model.predict(str(HERE/'risid/images'/image['file_name']),imgsz=640,device=0,conf=.001,iou=.7,max_det=300,half=False,verbose=False)[0]
            for box,score,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
                if int(cls)!=0: continue
                x,y,x2,y2=box
                current.append({'image_id':image['id'],'category_id':1,'bbox':[x,y,x2-x,y2-y],'score':score})
        write(chunk,current);predictions.extend(current)
        print(f'[RiSID {index+1}/20] {min(start+100,len(images))}/{len(images)} images',flush=True)
    write(out/'predictions.json',predictions)
    rows=metrics(evaluate(gt,predictions),evaluate(gt,predictions,.25))
    write(out/'metrics.json',rows)
    write(out/'COMPLETE.json',{'job':job,'images':len(images),'plastic_only':True,'smoke':smoke})

def export_external():
    rows=[]
    for job in read(HERE/'risid/plan.json')['jobs']:
        path=ROOT/'V2/risid_external/jobs'/Path(job['out']).name/'metrics.json'
        for row in read(path):
            rows.append({'model':job['spec']['id'],'seed':job['training']['seed'],**row})
    with (ROOT/'V2/risid_external/results_per_seed.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--external-index',type=int);p.add_argument('--smoke',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.external_index is not None: return external(a.external_index,a.smoke)
    plan=read(HERE/'risid/plan.json');pairs=read(ROOT/'V2/batch8_sensitivity/plan.json')['pairs']
    assert len(plan['jobs'])==20 and len(pairs)==10
    for job in plan['jobs']: assert (Path(job['out'])/'weights/best.pt').exists()
    for pair in pairs:
        old=read(Path(pair['batch4_out'])/'job.json')
        new=pair['batch8_job']
        assert {k:v for k,v in new['training'].items() if k!='batch'}=={k:v for k,v in old['training'].items() if k!='batch'}
        assert new['training']['batch']==8 and old['training']['batch']==4
    if a.check:
        print('OK: 20 external checkpoints; 10 paired batch-8 jobs; only physical batch changes.');return
    subprocess.run([sys.executable,str(HERE/'prepare_followup.py')],check=True)
    for i in range(len(plan['jobs'])):
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--external-index',str(i)],check=True)
    export_external()
    print('[RiSID COMPLETE] Starting batch-8 sensitivity.',flush=True)
    from run_experiment import train
    # TFE first to reveal potential memory failures before the remaining runs.
    for pair in sorted(pairs,key=lambda x:(x['batch8_job']['spec']['id']!='p2_tfe_off',x['batch8_job']['training']['seed'])):
        job=pair['batch8_job'];train(job)
        subprocess.run([sys.executable,'-u',str(HERE/'eval_worker.py'),'--job',str(Path(job['out'])/'job.json'),'--bootstrap','1000'],check=True)
    print('COMPLETE: RiSID and batch-8 training/evaluation. Statistical interpretation remains to be reviewed.',flush=True)

if __name__=='__main__': main()
