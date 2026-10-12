"""Verify the supplied archive or explicitly materialize relocated job configs."""
from pathlib import Path
import argparse,json,hashlib
R=Path(__file__).resolve().parent
def expand(x):
    if isinstance(x,dict):return {k:expand(v) for k,v in x.items()}
    if isinstance(x,list):return [expand(v) for v in x]
    return x.replace('${ROOT}',R.as_posix()) if isinstance(x,str) else x
def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--materialize',action='store_true');a=p.parse_args()
    hashes=json.loads((R/'SHA256SUMS.json').read_text())
    bad=[name for name,h in hashes.items() if not (R/name).is_file() or hashlib.sha256((R/name).read_bytes()).hexdigest()!=h]
    if bad:raise SystemExit('Integrity check failed: '+', '.join(bad[:10]))
    print(f'Integrity verified: {len(hashes)} files; no training started.')
    if not a.materialize:return
    import yaml
    for folder in ['revision/prepared','V2/mdpi_revision_supreme_v2_640','V2/batch8_sensitivity','revision/risid']:
        for f in (R/folder).rglob('*.json'):
            f.write_text(json.dumps(expand(json.loads(f.read_text(encoding='utf-8'))),indent=2),encoding='utf-8')
        for f in (R/folder).rglob('*.txt'):
            f.write_text(f.read_text().replace('${ROOT}',R.as_posix()))
    records=json.loads((R/'revision/prepared/manifest.json').read_text())
    for row in records:
        for field,sub in [('image','images'),('label','labels')]:
            row[field]=str(R/'revision/prepared/dataset'/row['split']/sub/Path(row[field]).name)
    (R/'revision/prepared/manifest.json').write_text(json.dumps(records,indent=2))
    data={'path':str(R/'revision/prepared/dataset'),'train':'train/images','val':'valid/images','test':'test/images','names':{0:'plastic',1:'debris'}}
    (R/'revision/prepared/data.yaml').write_text(yaml.safe_dump(data))
    for fold in (R/'revision/prepared').glob('cv_fold*'):
        d={'path':str(fold),'train':str(fold/'train.txt'),'val':str(fold/'valid.txt'),'test':str(fold/'test.txt'),'names':data['names']}
        (fold/'data.yaml').write_text(yaml.safe_dump(d))
    dest=R/'portable_jobs';dest.mkdir(exist_ok=True)
    jobs=expand(json.loads((R/'plans/training_jobs.json').read_text()))
    for i,j in enumerate(jobs):(dest/f'{i:04d}.json').write_text(json.dumps(j,indent=2))
    print(f'Materialized {len(jobs)} jobs. No training started. Hashes describe the original archive, before relocation.')
if __name__=='__main__':main()
