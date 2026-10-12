"""COCO 101-point AP with native-pixel strata and ignored out-of-stratum GT.

Keep ALL annotations. COCOeval areaRng marks other-scale GT ignored and ignores
their matches. Background FPs with areas in the active range remain counted.
Every image, including negative images, is retained in every area evaluation.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STRATA=['all','tiny','very_small','small','medium_large']
RANGES=[[0,1e12],[0,np.nextafter(256.,0.)],[256,np.nextafter(1024.,0.)],[1024,np.nextafter(4096.,0.)],[4096,1e12]]

def ground_truth(records, class_names=('plastic','debris')):
    data={'info':{},'images':[],'annotations':[],'categories':[{'id':i+1,'name':name} for i,name in enumerate(class_names)]}
    for i,r in enumerate(records,1):
        data['images'].append({'id':i,'file_name':Path(r['image']).name,'width':r['width'],'height':r['height']})
        for b in r['boxes']:
            x,y,w,h=b['xywhn'];w*=r['width'];h*=r['height'];x=x*r['width']-w/2;y=y*r['height']-h/2
            data['annotations'].append({'id':len(data['annotations'])+1,'image_id':i,'category_id':b['class']+1,'bbox':[x,y,w,h],'area':w*h,'iscrowd':0})
    return data

def evaluate(gt,predictions,confidence=None):
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO();coco.dataset=copy.deepcopy(gt);coco.createIndex()
        detections=predictions if confidence is None else [p for p in predictions if p['score']>=confidence]
        if detections:
            dt=coco.loadRes(detections)
        else:
            dt=COCO();dt.dataset={'info':{},'images':gt['images'],'categories':gt['categories'],'annotations':[]};dt.createIndex()
        e=COCOeval(coco,dt,'bbox');e.params.areaRng=RANGES;e.params.areaRngLbl=STRATA;e.params.maxDets=[1,10,300]
        e.params.iouThrs=np.r_[.25,np.linspace(.5,.95,10)];e.evaluate();e.accumulate()
    return e

def mean_valid(values):
    valid=values[values>=0]
    return float(valid.mean()) if len(valid) else None

def metrics(e, fixed=None):
    output=[]
    for a,scale in enumerate(STRATA):
        for k,name in [(None,'macro')]+[(i,c['name']) for i,c in enumerate(e.cocoGt.dataset['categories'])]:
            p=e.eval['precision'][:,:,:,a,-1]
            if k is not None:p=p[:,:,k:k+1]
            row={'stratum':scale,'class':name,'AP25':mean_valid(p[0]),'AP50':mean_valid(p[1]),'AP50_95':mean_valid(p[1:])}
            if fixed is not None:
                tp=fp=positives=0
                for item in fixed.evalImgs:
                    if item is None or item['aRng']!=RANGES[a] or (k is not None and item['category_id']!=e.cocoGt.dataset['categories'][k]['id']):continue
                    ignore=item['dtIgnore'][1]; matched=item['dtMatches'][1]>0
                    tp+=int(np.sum(matched & ~ignore));fp+=int(np.sum(~matched & ~ignore));positives+=int(np.sum(~np.asarray(item['gtIgnore'],dtype=bool)))
                row.update({'precision_conf025_iou050':tp/(tp+fp) if tp+fp else 0.,'recall_conf025_iou050':tp/positives if positives else None,'TP':tp,'FP':fp,'GT':positives})
            output.append(row)
    return output

def resample_evaluator(e, indices):
    """Reuse IoU matching; accumulate a multiset of independently sampled images."""
    sampled=copy.copy(e);sampled.params=copy.deepcopy(e.params)
    original_ids=e._paramsEval.imgIds; n=len(original_ids); k=len(e._paramsEval.catIds);a=len(RANGES)
    sampled.params.imgIds=list(range(1,len(indices)+1));sampled._paramsEval=copy.deepcopy(sampled.params)
    sampled.evalImgs=[e.evalImgs[cat*a*n+area*n+i] for cat in range(k) for area in range(a) for i in indices]
    sampled.eval={}
    with contextlib.redirect_stdout(io.StringIO()):sampled.accumulate()
    return sampled

def bootstrap(e, records, repeats=1000, seed=20260930, unit='frame'):
    groups={}
    for i,r in enumerate(records):groups.setdefault(r['frame'] if unit=='frame' else str(i),[]).append(i)
    keys=sorted(groups);rng=np.random.default_rng(seed);draws=[];draws50=[];draws25=[]
    for draw in range(repeats):
        if draw%100==0:print(f'[BOOTSTRAP {unit}] {draw}/{repeats}',flush=True)
        indices=[i for key in rng.choice(keys,len(keys),replace=True) for i in groups[key]]
        sampled=resample_evaluator(e,indices)
        summary=metrics(sampled)
        draws.append([m['AP50_95'] for m in summary]);draws50.append([m['AP50'] for m in summary]);draws25.append([m['AP25'] for m in summary])
    values=np.array([[np.nan if x is None else x for x in row] for row in draws])
    values50=np.array([[np.nan if x is None else x for x in row] for row in draws50]);values25=np.array([[np.nan if x is None else x for x in row] for row in draws25])
    ci=[]
    for i,m in enumerate(metrics(e)):
        v=values[:,i];v=v[np.isfinite(v)]
        extra={}
        for label,arr in [('AP25',values25),('AP50',values50)]:
            sample=arr[:,i];sample=sample[np.isfinite(sample)];extra.update({label+'_ci_low':float(np.quantile(sample,.025)) if len(sample) else None,label+'_ci_high':float(np.quantile(sample,.975)) if len(sample) else None})
        ci.append({**m,**extra,'ci_low':float(np.quantile(v,.025)) if len(v) else None,'ci_high':float(np.quantile(v,.975)) if len(v) else None,'valid_bootstrap_draws':len(v),'bootstrap_repeats':repeats,'bootstrap_unit':unit,'bootstrap_groups':len(keys)})
    return ci,{'AP50_95':values,'AP50':values50,'AP25':values25}

def predict(model,records,imgsz=768,device=0):
    predictions=[]
    # Per-image native shapes; stream avoids keeping GPU result tensors alive.
    for i,r in enumerate(records,1):
        if i==1 or i%20==0 or i==len(records):print(f'[PREDICT] image {i}/{len(records)}',flush=True)
        result=model.predict(source=r['image'],imgsz=imgsz,device=device,conf=.001,iou=.7,max_det=300,half=False,verbose=False)[0]
        boxes=result.boxes
        for xyxy,score,cls in zip(boxes.xyxy.cpu().numpy(),boxes.conf.cpu().numpy(),boxes.cls.cpu().numpy()):
            x1,y1,x2,y2=map(float,xyxy)
            predictions.append({'image_id':i,'category_id':int(cls)+1,'bbox':[x1,y1,x2-x1,y2-y1],'score':float(score)})
    return predictions

def evaluate_predictions(records,predictions,out,repeats=1000,class_names=('plastic','debris')):
    print('[METRICS] Computing COCO metrics',flush=True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);gt=ground_truth(records,class_names)
    e=evaluate(gt,predictions);fixed=evaluate(gt,predictions,.25)
    rows=metrics(e,fixed)
    (out/'ground_truth.json').write_text(json.dumps(gt),encoding='utf-8')
    (out/'predictions.json').write_text(json.dumps(predictions),encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if repeats:
        both=[]
        for unit in ('image','frame'):
            print(f'[BOOTSTRAP] Starting {unit}, {repeats} draws on CPU',flush=True)
            ci,draws=bootstrap(e,records,repeats,unit=unit);both.extend(ci)
            np.save(out/f'bootstrap_{unit}.npy',draws['AP50_95'])
            for metric,values in draws.items():np.save(out/f'bootstrap_{unit}_{metric}.npy',values)
        (out/'confidence_intervals.json').write_text(json.dumps(both,indent=2),encoding='utf-8')
    return rows
