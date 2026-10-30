"""Preserve released FloW test; create validation only from released training."""
import collections
import hashlib
import json
import random
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from PIL import Image
import yaml

HERE=Path(__file__).resolve().parent
OUT=HERE/'flow_training'
RAW=HERE/'flow/raw'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if (OUT/'audit.json').exists():
        print('Already prepared; refusing to overwrite split.');return
    archive=RAW/'FloW_IMG.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in z.infolist():
            target=(RAW/item.filename).resolve()
            assert target.is_relative_to(RAW.resolve()), 'Unsafe archive path'
        z.extractall(RAW)
    records=[];changes=[]
    for source_split in ('training','test'):
        base=RAW/'FloW_IMG'/source_split
        for xml in sorted((base/'annotations').glob('*.xml')):
            tree=ET.parse(xml).getroot()
            image=base/'images'/(xml.stem+'.jpg')
            with Image.open(image) as pic:
                w,h=pic.size
            assert (w,h)==(int(tree.findtext('size/width')),int(tree.findtext('size/height')))
            boxes=[]
            for i,obj in enumerate(tree.findall('object')):
                assert obj.findtext('name')=='bottle'
                assert int(obj.findtext('difficult',obj.findtext('Difficult','0')))==0
                coords=[float(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
                x1,y1,x2,y2=coords
                x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
                if x2<=x1 or y2<=y1:
                    changes.append({'xml':str(xml),'object':i,'action':'drop_no_visible_area'});continue
                if coords!=[x1,y1,x2,y2]:
                    changes.append({'xml':str(xml),'object':i,'action':'clip','before':coords,'after':[x1,y1,x2,y2]})
                boxes.append({'class':0,'xywhn':[(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h]})
            uid=source_split+'_'+xml.stem
            records.append({'original_image':str(image),'original_xml':str(xml),'original_split':source_split,'original_filename':tree.findtext('filename'),'source_folder':tree.findtext('folder'),'frame':uid,'uid':uid,'width':w,'height':h,'sha256':digest(image),'boxes':boxes})
    assert len(records)==2000
    training=[r for r in records if r['original_split']=='training']
    test=[r for r in records if r['original_split']=='test']
    assert len(training)==1200 and len(test)==800
    assert not {r['sha256'] for r in training}&{r['sha256'] for r in test}
    # Keep exact duplicate training images together in the derived inner split.
    groups=collections.defaultdict(list)
    for r in training: groups[r['sha256']].append(r)
    keys=sorted(groups);random.Random(20261028).shuffle(keys)
    val_keys=set();count=0
    for key in keys:
        if count+len(groups[key])<=240:
            val_keys.add(key);count+=len(groups[key])
    assert count==240
    OUT.mkdir(parents=True,exist_ok=True)
    for r in records:
        split=r['split']='test' if r['original_split']=='test' else 'val' if r['sha256'] in val_keys else 'train'
        img=OUT/'dataset'/split/'images'/(r['uid']+'.jpg')
        label=OUT/'dataset'/split/'labels'/(r['uid']+'.txt')
        img.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        if not img.exists():img.hardlink_to(r['original_image'])
        label.write_text(''.join('0 '+' '.join(f'{x:.12g}' for x in b['xywhn'])+'\n' for b in r['boxes']))
        r.update(image=str(img),label=str(label),label_sha256=digest(label))
    splits={}
    for split in ('train','val','test'):
        rr=[r for r in records if r['split']==split]
        splits[split]={'images':len(rr),'boxes':sum(len(r['boxes']) for r in rr),'empty_images':sum(not r['boxes'] for r in rr),'source_folder_counts':dict(collections.Counter(r['source_folder'] for r in rr))}
        (OUT/f'{split}.txt').write_text('\n'.join(r['image'] for r in rr)+'\n')
    for a,b in (('train','val'),('train','test'),('val','test')):
        assert not {r['sha256'] for r in records if r['split']==a}&{r['sha256'] for r in records if r['split']==b}
    (OUT/'data.yaml').write_text(yaml.safe_dump({'path':str(OUT/'dataset'),'train':'train/images','val':'val/images','test':'test/images','names':{0:'bottle'}}))
    (OUT/'manifest.json').write_text(json.dumps(records,indent=2))
    (OUT/'annotation_changes.json').write_text(json.dumps(changes,indent=2))
    audit={'archive_sha256':digest(archive),'source':'FloW-Img official training/test directory layout; downloaded by user','split_seed':20261028,'splits':splits,'boxes_in_download':sum(len(r['boxes']) for r in records),'changes':changes,'limitations':'Original split shares source-folder labels; temporal/site independence not established. Derived validation is random within official training, grouping identical file hashes. frame field is an image identifier, NOT a video group. Paper reports 5271 boxes; downloaded XML files contain 5272 before processing. VOC coordinates are interpreted directly as xyxy continuous bounds, no +/-1 adjustment. No test tuning.'}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2,ensure_ascii=True))

if __name__=='__main__': main()
