import json,glob,re,statistics as st
import os
R=os.environ.get('D3_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','results'))+'/'
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained(os.environ.get('D3LLM_MODEL','/mnt/nvme/models/d3llm/d3LLM_LLaDA'),trust_remote_code=True)
def rows(tag):
    f=glob.glob(R+tag+'/*/samples*.jsonl')[0]
    return [json.loads(l) for l in open(f)]
def fl(tag): return [r for r in rows(tag) if r['filter']=='flexible-extract']
def tgt(r): return r['target'].split('####')[-1].strip().replace(',','')
for tag in ['d3llm-fp16-kvcache','qwen-q4km-512tok']:
    rr=fl(tag)
    txt=[r['resps'][0][0] for r in rr]
    ntok=[len(tk(t,add_special_tokens=False)['input_ids']) for t in txt]
    ch=[len(t) for t in txt]
    print(tag,'LLaDA-tokenizer tokens/q mean(11-100)=%.1f mean(all)=%.1f; >=250: %d; chars/q mean(11-100)=%.1f median=%d'%(st.mean(ntok[10:]),st.mean(ntok),sum(n>=250 for n in ntok),st.mean(ch[10:]),st.median(ch)))
    wrong=[(i,r) for i,r in enumerate(rr) if not r['exact_match']]
    art=0
    for i,r in wrong:
        t=txt[i]; 
        if re.search(r'(?<![\d.])'+re.escape(tgt(r))+r'(?![\d])',t.replace(',','')[-150:]): art+=1
    print('  wrong:',len(wrong),'with target number in last 150 chars (possible extraction artefact):',art)
    # tail endings
    print('  ends with [boxed]:',sum('boxed' in t for t in txt),' median chars',st.median(ch))
