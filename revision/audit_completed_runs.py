"""Audit retained evidence, without interpreting absence of logs as no failures."""
from pathlib import Path
import csv, json, re, hashlib, datetime, collections
R=Path(__file__).resolve().parents[1]
O=R/'revision/manuscript_revision/supplement'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ''
def write(name,rows):
    with (O/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    mainjobs=read(R/'V2/mdpi_revision_supreme_v2_640/test_evaluation_plan.json')['jobs']
    cvjobs=[read(p) for p in (R/'V2/mdpi_revision_supreme_v2_640/jobs').glob('*cv*/job.json') if (p.parent/'TRAINING_COMPLETE.json').exists()]
    assert len(cvjobs)==75,len(cvjobs)
    batchjobs=[p['batch8_job'] for p in read(R/'V2/batch8_sensitivity/plan.json')['pairs']]
    rows=[];events=[]
    pattern=re.compile(r'Traceback \(most recent call last\)|CUDA out of memory|(?:RuntimeError|ValueError|KeyboardInterrupt|InterruptedError|MemoryError):')
    for scope,jobs in [('main',mainjobs),('grouped_cv',cvjobs),('batch8',batchjobs)]:
        for j in jobs:
            out=Path(j['out']);log=out/'console.log';actual=read(out/'actual_training.json')
            matches=[]
            for n,line in enumerate(log.read_text(encoding='utf-8',errors='replace').splitlines(),1) if log.exists() else []:
                if pattern.search(line):matches.append((n,line.strip()))
            for n,line in matches:events.append(dict(scope=scope,job=out.name,event='retained_log_error_signature',timestamp_utc='',source=log.relative_to(R).as_posix(),line=n,detail=line[:800]))
            resultrows=list(csv.DictReader((out/'results.csv').open(encoding='utf-8-sig')))
            evalmarker=out/'evaluation/metrics.json'
            rows.append(dict(scope=scope,job=out.name,training_complete=(out/'TRAINING_COMPLETE.json').exists(),stop_epoch=actual['stop_epoch'],result_epoch_rows=len(resultrows),best_checkpoint_present=(out/'weights/best.pt').exists(),metrics_present=evalmarker.exists(),console_present=log.exists(),error_signature_lines=len(matches),console_sha256=digest(log),actual_training_sha256=digest(out/'actual_training.json')))
    for j in read(R/'revision/risid/plan.json')['jobs']:
        out=R/'V2/risid_external/jobs'/Path(j['out']).name
        rows.append(dict(scope='external_inference',job=out.name,training_complete='not_applicable',stop_epoch='',result_epoch_rows='',best_checkpoint_present=Path(j['out'],'weights/best.pt').exists(),metrics_present=(out/'metrics.json').exists(),console_present=False,error_signature_lines='',console_sha256='',actual_training_sha256=digest(out/'COMPLETE.json')))
    for p in (R/'V2').rglob('failures.jsonl'):
        if 'mdpi_revision_supreme_v2_640' not in p.parts and 'batch8_sensitivity' not in p.parts and 'risid_external' not in p.parts:continue
        for n,line in enumerate(p.read_text().splitlines(),1):
            event=json.loads(line)
            events.append(dict(scope='campaign_unassigned',job='not_recorded',event=event['type'],timestamp_utc=datetime.datetime.fromtimestamp(event['timestamp'],datetime.timezone.utc).isoformat(),source=p.relative_to(R).as_posix(),line=n,detail=event.get('error','') or 'No job identifier or causal detail recorded. Cannot attribute to OOM or hardware.'))
    write('S14_completion_and_log_audit.csv',rows)
    write('S15_retained_incident_register.csv',events)
    summary={'scope_counts':dict(collections.Counter(r['scope'] for r in rows)),'all_235_training_stop_epoch_150':all(r['stop_epoch']==150 for r in rows if r['scope']!='external_inference'),'training_console_logs':sum(r['console_present'] for r in rows),'error_signature_lines':sum(r['error_signature_lines'] for r in rows if isinstance(r['error_signature_lines'],int)),'recorded_events':events,'limits':'Retained log audit, not exhaustive operational history. Logs are append-mode but external console transcripts and OS/hardware telemetry were not retained. An unassigned KeyboardInterrupt cannot establish its cause or affected job. Completion is not proof of uninterrupted execution.'}
    (R/'revision/fixed_pr_evidence/incident_audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
