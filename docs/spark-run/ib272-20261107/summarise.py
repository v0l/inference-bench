import json,glob,re,statistics as st,sys
d=sys.argv[1] if len(sys.argv) > 1 else '.'
for m in ['qwen3','qwen25']:
  for n in [4,16]:
    r={}
    r['vllm']=[ [x for x in json.load(open(f)) if x['batch']==n][0]['decode_tok_s'] for f in glob.glob(f'{d}/{m}-vllm-*.json')]
    r['llama']=[]
    for f in glob.glob(f'{d}/{m}-llama-*.jsonl'):
      for l in open(f):
        j=json.loads(l)
        if j.get('pl')==n: r['llama'].append(j['speed_tg'])
    for e in ['base','new']:
      r[e]=[]
      for f in glob.glob(f'{d}/{m}-{e}-*.txt'):
        for l in open(f):
          mm=re.search(r'batch=(\d+).*decode=([\d.]+)',l)
          if mm and int(mm[1])==n: r[e].append(float(mm[2]))
    print(m,n,{k:(round(st.median(v),1),len(v)) for k,v in r.items()})
