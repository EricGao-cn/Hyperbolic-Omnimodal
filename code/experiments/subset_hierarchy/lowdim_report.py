"""Export only verified completed runs; missing/failed runs remain explicit."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.subset_hierarchy.lowdim_common import *


def flatten(value,prefix=''):
    out={}
    for key,item in value.items():
        key=prefix+str(key)
        if isinstance(item,dict):out.update(flatten(item,key+'.'))
        elif isinstance(item,(int,float,str,bool)) or item is None:out[key]=item
    return out


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def collect(roots):
    suite=roots['suite'];records=[];geometry=[];metrics={}
    source=digest(source_manifest(roots));protocol=file_hash(suite/'protocol.json')
    for task in read_json(suite/'tasks.json'):
        run=suite/'runs'/task['name'];row=dict(task,status='PENDING',seed_count=1)
        if (run/'DONE.json').exists():
            try:
                done=read_json(run/'DONE.json');metric=read_json(run/'test_metrics.json');config=metric['config']
                if not done['passed'] or done['epochs']!=10 or metric['epochs_completed']!=10:raise ValueError('Incomplete training budget')
                if config['protocol_sha256']!=protocol or config['source']!=source:raise ValueError('Source/protocol mismatch')
                if config['data']!=read_json(suite/'data'/task['dataset']/'READY.json'):raise ValueError('Data cohort mismatch')
                row.update(status='COMPLETE',selected_epoch=metric['selected_epoch'],steps=metric['steps'],world_size=config['world_size'],**flatten(metric['macro'],'macro.'))
                for direction,values in metric['directions'].items():row.update(flatten(values,direction+'.'))
                if (run/'epochs.jsonl').exists():row['training_and_validation_seconds']=sum(json.loads(line)['elapsed'] for line in (run/'epochs.jsonl').read_text().splitlines())
                if (run/'steps.jsonl').exists():
                    steps=[json.loads(line) for line in (run/'steps.jsonl').read_text().splitlines()]
                    row['peak_allocated_gib']=max(s['peak_allocated_gib'] for s in steps)
                    row['peak_reserved_gib']=max(s['peak_reserved_gib'] for s in steps)
                    row['median_logged_samples_per_second']=64/statistics.median(s['seconds_per_batch'] for s in steps)
                if 'geometry' in metric:geometry.append(dict(task,**flatten(metric['geometry'])))
                metrics[task['name']]=metric
            except (KeyError,ValueError,FileNotFoundError) as error:row.update(status='INVALID',error=str(error))
        elif (run/'last.pt').exists():row['status']='INCOMPLETE_RESUMABLE'
        elif (roots['logs']/task['name']).exists() and list((roots['logs']/task['name']).glob('train_failure_*.json')):row['status']='FAILED'
        records.append(row)
    reference={(r['dataset'],r['method']):r['macro.R@1'] for r in records if r['status']=='COMPLETE' and r['dimension']==512}
    for row in records:
        ref=reference.get((row['dataset'],row['method']))
        if row['status']=='COMPLETE' and ref is not None:
            row['delta_R1_vs_new512']=row['macro.R@1']-ref
            row['retention_pct_vs_new512']=100*row['macro.R@1']/ref if ref else None
    return records,geometry,metrics


def plots(destination,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    done=[r for r in records if r['status']=='COMPLETE']
    if not done:return
    for dataset in DATASETS:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for method in METHODS:
            rows=sorted([r for r in done if r['dataset']==dataset and r['method']==method],key=lambda r:r['dimension'])
            if not rows:continue
            axes[0].plot([r['dimension'] for r in rows],[r['macro.R@1'] for r in rows],marker='o',label=method)
            retained=[r for r in rows if r.get('retention_pct_vs_new512') is not None]
            if retained:axes[1].plot([r['dimension'] for r in retained],[r['retention_pct_vs_new512'] for r in retained],marker='o',label=method)
        for ax in axes:ax.set_xscale('log',base=2);ax.set_xticks(DIMENSIONS,[str(d) for d in DIMENSIONS]);ax.set_xlabel('Embedding dimension');ax.grid(alpha=.2)
        axes[0].set_ylabel('Six-direction macro R@1 (%)');axes[1].set_ylabel('Retention vs matched new 512D (%)')
        axes[0].legend(fontsize=7);fig.suptitle(dataset+' — seed 50 only; missing runs omitted; no seed confidence intervals');fig.tight_layout()
        fig.savefig(destination/(dataset+'_dimensions.pdf'));fig.savefig(destination/(dataset+'_dimensions.png'),dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser();roots_args(p);p.add_argument('--plots',action='store_true');args=p.parse_args();roots=get_roots(args)
    destination=roots['suite']/'reports';destination.mkdir(parents=True,exist_ok=True)
    records,geometry,metrics=collect(roots)
    write_csv(destination/'retrieval.csv',records);write_csv(destination/'geometry.csv',geometry)
    write_json(destination/'results.json',dict(single_seed=True,seed=50,comparison='Matched newly trained 512D within each method and dataset',runs=records,metrics=metrics))
    summary={status:sum(r['status']==status for r in records) for status in sorted({r['status'] for r in records})}
    write_json(destination/'status.json',summary)
    if args.plots:plots(destination,records)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
