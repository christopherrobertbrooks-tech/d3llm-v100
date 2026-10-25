import json,glob,re
import os
R=os.environ.get('D3_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','results'))+'/'
def fl(tag):
    f=glob.glob(R+tag+'/*/samples*.jsonl')[0]
    return [r for r in map(json.loads,open(f)) if r['filter']=='flexible-extract']
def tgt(r): return r['target'].split('####')[-1].strip().replace(',','')
for tag in ['qwen-q4km-512tok','d3llm-fp16-kvcache']:
    rr=fl(tag); k=0
    print('#####',tag)
    for i,r in enumerate(rr):
        if r['exact_match']: continue
        t=r['resps'][0][0]
        if re.search(r'(?<![\d.])'+re.escape(tgt(r))+r'(?![\d])',t.replace(',','')[-150:]):
            k+=1
            if k<=4: print(f'--- doc {i} target {tgt(r)} filtered={r["filtered_resps"]}\n...{t[-260:]!r}')
