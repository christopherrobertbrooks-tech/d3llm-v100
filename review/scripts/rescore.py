import json,glob,re
import os
R=os.environ.get('D3_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','results'))+'/'
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained(os.environ.get('D3LLM_MODEL','/mnt/nvme/models/d3llm/d3LLM_LLaDA'),trust_remote_code=True)
def fl(tag):
    f=glob.glob(R+tag+'/*/samples*.jsonl')[0]
    return [r for r in map(json.loads,open(f)) if r['filter']=='flexible-extract']
def tgt(r): return r['target'].split('####')[-1].strip().replace(',','')
def nums(s): return [x.rstrip('.') for x in re.findall(r'-?\d[\d,]*\.?\d*',s.replace(',',''))]
def final_ok(t,r):
    m=re.findall(r'boxed\{([^}]*)\}',t)
    if m: return tgt(r) in nums(m[-1])
    last=[l for l in t.strip().split('\n') if l.strip()][-1]
    ns=nums(last)
    # last sentence only
    sent=re.split(r'(?<=[.!?])\s+',last.strip())[-1]
    return tgt(r) in nums(sent) or tgt(r) in ns
for tag in ['d3llm-fp16-kvcache','d3llm-bf16-kvcache','d3llm-fp16-multiblock','qwen-q4km-512tok','qwen-q4km']:
    rr=fl(tag)
    f=[final_ok(r['resps'][0][0],r) for r in rr]
    either=[a or bool(r['exact_match']) for a,r in zip(f,rr)]
    s=f"{tag}: lm_eval flexible={sum(r['exact_match'] for r in rr):.0f}  final-sentence/boxed rescoring={sum(f)}  union(lm_eval or final)={sum(either)}"
    if tag.startswith('d3llm-fp16-kv'):
        nt=[len(tk(r['resps'][0][0],add_special_tokens=False)['input_ids']) for r in rr]
        cap=[i for i,n in enumerate(nt) if n>=250]
        s+=f"\n   d3LLM resp >=250 LLaDA tokens: {len(cap)}; lm_eval-correct among them {sum(rr[i]['exact_match'] for i in cap):.0f}; have boxed {sum('boxed' in rr[i]['resps'][0][0] for i in cap)}; max tokens {max(nt)}"
        print(s)
        print('   hist of tokens:', sorted(nt)[-45:])
    else: print(s)
