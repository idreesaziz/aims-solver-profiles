"""Disk-backed reproduction of Drift v2's released prior, without retuning.

Uses original feature functions and same mathematical objective, item cap,
selection RNG, ridge penalties, weight exponent and L-BFGS settings. Streaming
changes floating-point reduction order; bit-identical historical weights are not
promised. Public pinned data only. No APIs, hidden data, uploads or credentials.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json,sqlite3,gc,importlib.util,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.optimize import fmin_l_bfgs_b
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
UP=ROOT/'external/aims-team-drift';sys.path.insert(0,str(UP/'submission'))
import featurize as F
import hashlib
def save(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
OUT=Path(os.environ.get('AIMS_PRIOR_DIR',str(ROOT/'work/prior')));DATA=Path(os.environ.get('AIMS_DATA_DIR',str(ROOT/'data/measurement-db')))
SOURCE=json.loads((UP/'report/TRAINING_SOURCES.json').read_text())
REV=next(iter(SOURCE['download_metadata_revisions_for_selected_tables']))
FIELDS=['normalized_name','provider','release_date','access_date','harness','harness_version','reasoning_effort','subject_features_extra']

def clean(x,limit):
    return '' if x is None or (isinstance(x,float) and np.isnan(x)) else str(x)[:limit]

def download():
    from huggingface_hub import hf_hub_download
    for b in SOURCE['benchmark_names']:
        for name in ['benchmarks','subjects','items','responses']:
            target=DATA/b/'formatted_tables'/f'{name}.parquet'
            if not target.exists():
                print('DOWNLOAD',b,name,flush=True)
                hf_hub_download(repo_id='aims-foundations/measurement-db',repo_type='dataset',revision=REV,
                    filename=f'{b}/formatted_tables/{name}.parquet',local_dir=str(DATA))

def write_csr(path,x):
    path.mkdir(parents=True,exist_ok=True)
    for k,v in [('data',x.data),('indices',x.indices),('indptr',x.indptr),('shape',np.array(x.shape))]:np.save(path/(k+'.npy'),v)

def csr(path):
    shape=tuple(np.load(path/'shape.npy'));a=[np.load(path/(k+'.npy'),mmap_mode='r') for k in ['data','indices','indptr']]
    return sparse.csr_matrix(tuple(a),shape=shape,copy=False)

def prepare(b):
    dst=OUT/'features'/b
    if (dst/'receipt.json').exists():return
    folder=DATA/b/'formatted_tables';dst.mkdir(parents=True,exist_ok=True)
    meta=pq.read_table(folder/'benchmarks.parquet').to_pylist()[0]
    assert meta['release']=='public',b
    # First pass establishes exactly upstream's observed, sorted subject/item IDs.
    item_ids=set();subject_ids=set()
    for batch in pq.ParquetFile(folder/'responses.parquet').iter_batches(batch_size=65536,columns=['subject_id','item_id','response']):
        df=batch.to_pandas().dropna(subset=['response']);assert df.response.isin([0,1]).all(),b
        item_ids.update(df.item_id.unique());subject_ids.update(df.subject_id.unique())
    items=sorted(item_ids);subjects=sorted(subject_ids);assert items and subjects,b
    rng=np.random.default_rng(int.from_bytes(b.encode()[:4].ljust(4,b'0'),'big'))
    keep=np.sort(rng.permutation(len(items))[:10000]);selected={items[i]:j for j,i in enumerate(keep)}
    sm={s:i for i,s in enumerate(subjects)}
    # Aggregate only selected items. SQLite bounds RAM even for dense panels.
    db=dst/'counts.sqlite';db.unlink(missing_ok=True)
    con=sqlite3.connect(db);con.execute('pragma journal_mode=off');con.execute('pragma synchronous=off')
    con.execute('create table counts(s integer,i integer,n integer,k integer,inter text,primary key(s,i)) without rowid')
    for batch in pq.ParquetFile(folder/'responses.parquet').iter_batches(batch_size=65536,columns=['subject_id','item_id','response','interactors']):
        df=batch.to_pandas().dropna(subset=['response']);df=df[df.item_id.isin(selected)]
        if df.empty:continue
        agg=df.groupby(['subject_id','item_id'],sort=True).agg(n=('response','size'),k=('response','sum'),inter=('interactors','first'))
        values=[(sm[s],selected[i],int(row['n']),int(row['k']),None if pd.isna(row['inter']) else str(row['inter'])) for (s,i),row in agg.iterrows()]
        con.executemany('insert into counts values(?,?,?,?,?) on conflict(s,i) do update set n=n+excluded.n,k=k+excluded.k,inter=coalesce(counts.inter,excluded.inter)',values);con.commit()
    # Upstream chooses first truthy interactor in sorted (subject,item) rows.
    inter={}
    for s,i,n,k,it in con.execute('select * from counts order by s,i'):
        if i not in inter and it:inter[i]=clean(it,300)
    count=con.execute('select count(*) from counts').fetchone()[0]
    arrays={k:np.lib.format.open_memmap(dst/(k+'.npy'),mode='w+',dtype=np.int32,shape=(count,)) for k in ['si','ii','n','k']}
    for at,(s,i,n,k,it) in enumerate(con.execute('select * from counts order by s,i')):
        for key,value in [('si',s),('ii',i),('n',n),('k',k)]:arrays[key][at]=value
    for a in arrays.values():a.flush()
    del arrays;con.close();db.unlink()
    subject_table={s['subject_id']:s for s in pq.read_table(folder/'subjects.parquet').to_pylist()}
    visible=[{k:clean(subject_table[s].get(k),400) for k in FIELDS} for s in subjects]
    save(dst/'subjects.json',visible)
    indices=[];ptr=[0]
    for s in visible:indices.extend(F.subject_idx(s));ptr.append(len(indices))
    sx=sparse.csr_matrix((np.ones(len(indices),np.float32),np.array(indices,np.int32),np.array(ptr)),shape=(len(visible),F.D_SUBJ));write_csr(dst/'S',sx)
    texts={}
    for batch in pq.ParquetFile(folder/'items.parquet').iter_batches(batch_size=16384,columns=['item_id','content','item_features']):
        df=batch.to_pandas();df=df[df.item_id.isin(selected)]
        for _,r in df.iterrows():
            j=selected[r.item_id]
            if j not in texts:texts[j]={'item_content':clean(r.content,3000),'item_features':clean(r.item_features,600),'interactors':inter.get(j,'')}
    ind=[];dat=[];ptr=[0]
    for j in range(len(keep)):
        it=texts.get(j,{'item_content':'','item_features':'','interactors':inter.get(j,'')})
        tf=F.item_tf(it);ind.extend(tf);dat.extend(tf.values());ptr.append(len(ind))
    x=sparse.csr_matrix((np.array(dat,np.float32),np.array(ind,np.int32),np.array(ptr)),shape=(len(keep),F.D_ITEM));write_csr(dst/'X',x)
    save(dst/'receipt.json',dict(benchmark=b,observed_items=len(items),selected_items=len(keep),subjects=len(subjects),pairs=count,
        response_sha=sha(folder/'responses.parquet'),items_sha=sha(folder/'items.parquet'),subjects_sha=sha(folder/'subjects.parquet')))
    print('PREPARED',b,len(keep),count,flush=True);gc.collect()

def load(b,normalized=None):
    d=OUT/'features'/b
    return dict(bench=b,X=csr(d/'X') if normalized is None else csr(normalized/b),S=csr(d/'S'),
                **{k:np.load(d/(k+'.npy'),mmap_mode='r') for k in ['si','ii','n','k']})

def fit_stream(bs,tag):
    dst=OUT/tag;dst.mkdir(parents=True,exist_ok=True)
    if (dst/'fit_receipt.json').exists():return
    norm=OUT/(tag+'_normalized');df=np.zeros(F.D_ITEM);N=0;Wn=0.;ntot=ktot=0
    for b in bs:
        f=load(b);df+=np.bincount(f['X'].indices,minlength=F.D_ITEM);N+=f['X'].shape[0]
        total=float(f['n'].sum());Wn+=np.sqrt(total);ntot+=total;ktot+=float(f['k'].sum())
    idf=(np.log((N+1)/(df+1))+1).astype(np.float32)
    for b in bs:
        f=load(b);x=f['X'].multiply(idf[None,:]).tocsr();nrm=np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())+1e-9
        x=sparse.diags(1/nrm).dot(x).tocsr().astype(np.float32);write_csr(norm/b,x)
    calls=0;t0=time.monotonic();sd=F.D_SUBJ;nd=F.D_ITEM
    def objective(theta):
        nonlocal calls
        b0=theta[0];ws=theta[1:1+sd];wi=theta[1+sd:1+sd+nd];loss=0.;gb=0.;gs=np.zeros(sd);gi=np.zeros(nd)
        for b in bs:
            f=load(b,norm);x=f['X'];s=f['S'];n=f['n'].astype(float);k=f['k'].astype(float)
            wt=float(n.sum())**(-.5);z=(s@ws)[f['si']]+(x@wi)[f['ii']]+b0
            p=1/(1+np.exp(-np.clip(z,-30,30)));r=wt*(n*p-k)
            loss+=wt*(n@np.logaddexp(0,z)-k@z);gb+=r.sum()
            gs+=s.T@np.bincount(f['si'],weights=r,minlength=s.shape[0]);gi+=x.T@np.bincount(f['ii'],weights=r,minlength=x.shape[0])
        value=loss/Wn+.005*(ws@ws+wi@wi)
        # Original trainer includes one unused zero-gradient slot per benchmark.
        grad=np.r_[gb/Wn,gs/Wn+.01*ws,gi/Wn+.01*wi,np.zeros(len(bs))]
        calls+=1
        save(dst/'status.json',dict(calls=calls,objective=float(value),seconds=time.monotonic()-t0))
        print('OPT',tag,calls,round(float(value),9),round(time.monotonic()-t0,1),flush=True)
        return value,grad
    theta=np.zeros(1+sd+nd+len(bs));p=ktot/ntot;theta[0]=np.log(p/(1-p))
    th,value,info=fmin_l_bfgs_b(objective,theta,maxiter=120,m=20)
    save(dst/'meta.json',dict(b=float(th[0]),lam_i=.01,lam_s=.01,alpha=.5,n_benchmarks=len(bs)))
    for key,a in [('ws',th[1:1+sd]),('wi',th[1+sd:1+sd+nd]),('idf',idf)]:a.astype('<f4').tofile(dst/(key+'.f32'))
    save(dst/'fit_receipt.json',dict(benchmarks=bs,objective=float(value),iterations=int(info['nit']),warnflag=int(info['warnflag']),task=str(info['task']),seconds=time.monotonic()-t0,
        caveat='Same objective; streaming reduction order differs. Not historical weights.',hashes={k:sha(dst/k) for k in ['meta.json','ws.f32','wi.f32','idf.f32']}))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'plan.json',dict(revision=REV,benchmarks=SOURCE['benchmark_names'],ridge_subject=.01,ridge_item=.01,alpha=.5,max_items=10000,
        original_trainer_sha=sha(UP/'dev/train_prior.py'),original_features_sha=sha(UP/'submission/featurize.py'),script_sha=sha(__file__),paid_calls=0,submission_changed=False))
    download()
    for b in SOURCE['benchmark_names']:prepare(b)
    # Matched cold-bank diagnostic keeps direct and multi-benchmark core sources
    # out. Final full artifact is separate and must NEVER be evaluated on those.
    excluded=['mmlupro','fewshot_ttt_bbh','llm_uncertainty_bench','naturalreasoning','robust_reasoning_benchmark']
    fit_stream([b for b in SOURCE['benchmark_names'] if b not in excluded],'cold_core')
    fit_stream(SOURCE['benchmark_names'],'full_public')
    print('DONE',flush=True)
if __name__=='__main__':main()
