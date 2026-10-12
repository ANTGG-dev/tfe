import argparse
import json
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--bootstrap',type=int,default=1000);a=p.parse_args()
    job=json.loads(Path(a.job).read_text())
    if job['spec'].get('backend_path'):sys.path.insert(0,job['spec']['backend_path'])
    from run_experiment import evaluate_job
    evaluate_job(job,a.bootstrap)

if __name__=='__main__':main()
