import json,glob,re,statistics as st,sys
import os
R=os.environ.get('D3_RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','results'))+'/'
def samples(tag):
    f=glob.glob(R+tag+'/*/samples*.jsonl')[0]
    rows=[json.loads(l) for l in open(f)]
    return rows
def acc(rows):
    out={}
    for r in rows:
        out.setdefault(r['filter'],[]).append(r)
    return out
def num(s):
    return s.replace(',','').replace('$','').rstrip('.')
def last_number(t):
    m=re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)',t)
    if not m: return None
    m=m[-1]; return [x for x in m if x][0]
def tgt(r): return r['target'].split('####')[-1].strip().replace(',','')
def correct(resp,r):
    n=last_number(resp)
    return n is not None and num(n).lower()==tgt(r)
def parse_d3(tag):
    log=open(R+tag+'.log',errors='replace').read()
    nfes=[int(x) for x in re.findall(r'NFE \(Number of Function Evaluations\): (\d+)',log)]
    tt=[float(x) for x in re.findall(r'Total time taken \(after warmup\): ([\d.]+)',log)]
    tn=[int(x) for x in re.findall(r'Total NFE is (\d+)',log)]
    tk=[int(x) for x in re.findall(r'Total tokens: (\d+)',log)]
    thr=re.findall(r'Throughput: ([\d.]+)',log)
    gm=[float(x) for x in re.findall(r'GPU Memory Used: ([\d.]+) MB',log)]
    return dict(nfes=nfes,tt=tt,tn=tn,tk=tk,thr=thr,gm=gm,log=log)
res={}
for tag in ['d3llm-fp16-kvcache','d3llm-bf16-kvcache','d3llm-fp16-multiblock']:
    rows=samples(tag); a=acc(rows)
    p=parse_d3(tag)
    fl=a['flexible-extract']; sm=a['strict-match']
    k=sum(r['exact_match'] for r in fl); ks=sum(r['exact_match'] for r in sm)
    mine=[correct(r['resps'][0][0],r) for r in fl]
    n=len(fl); pr=k/n; se=(pr*(1-pr)/(n-1))**.5
    # per-request times from cumulative totals
    tt=p['tt']; per=[tt[0]]+[tt[i]-tt[i-1] for i in range(1,len(tt))]
    nf=p['nfes']
    print(f"== {tag}: n={n} flex={k} strict={ks} mine_lastnum={sum(mine)} (agree {sum(m==bool(r['exact_match']) for m,r in zip(mine,fl))}/{n}) stderr~{se:.4f}")
    print(f"   requests logged NFE={len(nf)} cumul-time lines={len(tt)} final: time={tt[-1]:.2f}s NFE={p['tn'][-1]} tokens={p['tk'][-1]} thr(calc)={p['tk'][-1]/tt[-1]:.2f} tok/fwd={p['tk'][-1]/p['tn'][-1]:.3f} logged thr={p['thr'][-1]}")
    post=nf[10:]
    print(f"   sum NFE all={sum(nf)} sum NFE after warmup (10:)={sum(post)} vs logged Total NFE {p['tn'][-1]}; s/q={tt[-1]/len(tt):.3f}; ms/forward(after-warmup)={1000*tt[-1]/p['tn'][-1]:.1f}; ms/fwd incl all NFE={1000*tt[-1]/sum(nf):.1f}")
    print(f"   per-request s: median={st.median(per):.2f} min={min(per):.2f} max={max(per):.2f}; first-10 NFE mean={st.mean(nf[:10]):.1f} rest={st.mean(post):.1f}")
    print(f"   GPU mem used print (avg over 2 GPUs, MB): min={min(p['gm']):.0f} max={max(p['gm']):.0f}")
    lens=[len(r['resps'][0][0]) for r in fl]
    nan=sum(1 for r in fl if 'nan' in r['resps'][0][0].lower().split())
    emp=sum(1 for r in fl if len(r['resps'][0][0].strip())<20)
    print(f"   resp chars median={st.median(lens)} empty/short(<20)={emp} 'nan' tokens={nan}; invalid-strict={sum(1 for r in sm if r['filtered_resps']==['[invalid]'])}")
    res[tag]=dict(rows=fl,mine=mine,nf=nf)
# fp16 vs bf16 agreement
a,b=res['d3llm-fp16-kvcache'],res['d3llm-bf16-kvcache']
same=sum(x['resps'][0][0]==y['resps'][0][0] for x,y in zip(a['rows'],b['rows']))
sameans=sum(last_number(x['resps'][0][0])==last_number(y['resps'][0][0]) for x,y in zip(a['rows'],b['rows']))
both=sum(m and n for m,n in zip(a['mine'],b['mine'])); onlyf=sum(m and not n for m,n in zip(a['mine'],b['mine'])); onlyb=sum(n and not m for m,n in zip(a['mine'],b['mine']))
print(f"fp16 vs bf16: identical text {same}/100, identical extracted answer {sameans}/100; both right {both}, fp16 only {onlyf}, bf16 only {onlyb}; NFE equal {sum(x==y for x,y in zip(a['nf'],b['nf']))}")
c=res['d3llm-fp16-multiblock']
print("fp16 kv vs nocache identical text", sum(x['resps'][0][0]==y['resps'][0][0] for x,y in zip(a['rows'],c['rows'])), "right: both",sum(m and n for m,n in zip(a['mine'],c['mine'])))
# Qwen
def parse_server(tag):
    s=open(R+tag+'-server.log',errors='replace').read()
    pe=re.findall(r'prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens',s)
    ev=re.findall(r'\|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens',s)
    tot=re.findall(r'total time =\s+([\d.]+) ms /\s+(\d+) tokens',s)
    trunc=re.findall(r'truncated = (\d)',s)
    return [(float(a),int(b)) for a,b in pe],[(float(a),int(b)) for a,b in ev],[(float(a),int(b)) for a,b in tot],trunc,s
for tag,mx in [('qwen-q4km',256),('qwen-q4km-512tok',512)]:
    rows=samples(tag); a=acc(rows); fl=a['flexible-extract']
    k=sum(r['exact_match'] for r in fl); mine=[correct(r['resps'][0][0],r) for r in fl]
    pe,ev,tot,trunc,s=parse_server(tag)
    print(f"== {tag}: n={len(fl)} flex={k} strict={sum(r['exact_match'] for r in a['strict-match'])} mine={sum(mine)}; server requests: prompt={len(pe)} eval={len(ev)} total={len(tot)}")
    if len(tot)==100:
        # skip first 10 like claim
        def agg(lo):
            T=sum(x[0] for x in tot[lo:])/1000; G=sum(x[1] for x in ev[lo:]); Ptk=sum(x[1] for x in pe[lo:]); Gt=sum(x[0] for x in ev[lo:])/1000
            return T,G,Ptk,Gt
        for lo in (10,0):
            T,G,Ptk,Gt=agg(lo)
            n=100-lo
            print(f"   reqs {lo+1}-100: total_time={T:.2f}s gen_tokens={G} ({G/n:.1f}/q) prompt_tokens={Ptk}; tok/s(total time)={G/T:.2f}; tok/s(gen only)={G/Gt:.2f}; s/q={T/n:.3f}")
        gens=[x[1] for x in ev]
        cut=sum(1 for g in gens if g>=mx)
        print(f"   responses hitting max_gen_toks {mx}: {cut}; median gen={st.median(gens)}; truncated flags={trunc.count('1')}")
        # accuracy among cut vs not
        cutidx=[i for i,g in enumerate(gens) if g>=mx]
        print(f"   correct among cut: {sum(mine[i] for i in cutidx)}/{len(cutidx)}; among not cut: {sum(mine[i] for i in range(100) if i not in cutidx)}/{100-len(cutidx)}")
        # check correspondence order: samples order doc_id vs server order
    # presence of final-answer patterns
    fl_txt=[r['resps'][0][0] for r in fl]
    print("   resps containing 'answer is' or boxed:",sum(1 for t in fl_txt if re.search(r'answer is|boxed|\*\*Answer',t,re.I)))
