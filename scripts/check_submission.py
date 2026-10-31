"""Exact-ZIP dispatch parity, isolation and evidence-boundary checks."""
import json,subprocess,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CHILD=r'''
import builtins,socket,sys,time,json
from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root))
original=builtins.__import__
def guarded(name,*a,**k):
    if name.split('.')[0] in {'torch','transformers','sklearn','google','requests'}:raise ImportError(name)
    return original(name,*a,**k)
builtins.__import__=guarded
def blocked(*a,**k):raise AssertionError('Network attempted')
socket.socket.connect=blocked;socket.create_connection=blocked
import numpy as np
import model
from models.subject_adaptation_v79 import Adaptation
assert Path(model.__file__).resolve().is_relative_to(root.resolve())
def subj(j):return dict(normalized_name='solver'+str(j),provider='Test',release_date='',access_date='',harness='h',harness_version='',reasoning_effort='',subject_features_extra='')
def item(j,bid='b',tool=''):return dict(item_content='Synthetic arithmetic '+str(j)+': What is 3+4?',item_features='',interactors=tool,benchmark_id=bid)
p=model._get();maximum=0.;checks=0;started=time.monotonic()
for size in [1,3,10]:
    subjects=[subj(j) for j in range(size)];items=[item(i) for i in range(45)]
    p0=np.array([[p.probability(s,i) for i in items] for s in subjects])
    for budget in [0,1,3,7,15,31]:
        support=[np.arange(budget) for _ in subjects];ys=[np.array([(i+j)%2 for i in range(budget)],float) for j in range(size)]
        L=[[[s,items[int(i)]],int(y)] for s,ii,yy in zip(subjects,support,ys) for i,y in zip(ii,yy)]
        ref=Adaptation(p0,support,ys,.25,'pseudo4')
        for j,s in enumerate(subjects):
            for q in [33,34]:
                expected=ref.predict(p0[j,[q]],np.array([q]),j)[0];actual=model.predict([s,items[q]],L)
                maximum=max(maximum,abs(expected-actual));checks+=1
                assert abs(expected-actual)<1e-11,(size,budget,j,q,expected,actual)
target=[subj(0),item(40)];L=[[[subj(0),item(0)],1],[[subj(1),item(0)],0],[[subj(1),item(1)],1]]
v=model.predict(target,L)
assert v==model.predict(target,L+[[[subj(0),item(5,'unrelated')],0]])
assert v==model.predict(target,L)
# Same-subject fallback when benchmark IDs are unavailable; foreign subject
# evidence must not influence an unscoped prediction.
t=[subj(0),item(40,'')];same=[[[subj(0),item(0,'')],1]]
assert model.predict(t,same)==model.predict(t,same+[[[subj(1),item(1,'')],0]])
# Unequal counts, repeated outcomes, tool-distinct items, unseen target subject.
weird=[[[subj(0),item(0)],1],[[subj(0),item(0)],0],[[subj(2),item(0,tool='browser')],1]]
for j in [0,1,2,9]:
    v=model.predict([subj(j),item(0,tool='browser' if j==2 else '')],weird)
    assert type(v) is float and 0<=v<=1
print(json.dumps(dict(passed=True,checks=checks,max_error=maximum,network_blocked=True,deep_model_imports_blocked=True,
    benchmark_isolation=True,missing_benchmark_isolation=True,unequal_counts_repeated_trials_tools_unseen_subject=True,seconds=time.monotonic()-started)))
'''
def main():
    archive=ROOT/'submission/submission_drift_prior_ours_v82.zip'
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith(('.py','.json','.txt','.md')):
                data=z.read(name)
                assert not any(k in data for k in [b'"refresh_token"',b'"client_secret"',b'-----BEGIN PRIVATE KEY-----']),name
        with tempfile.TemporaryDirectory(prefix='aims-v82-') as tmp:
            tmp=Path(tmp);dest=tmp/'submission';z.extractall(dest)
            child=tmp/'check.py';child.write_text(CHILD)
            proc=subprocess.run([sys.executable,'-I',str(child),str(dest)],capture_output=True,text=True,cwd=tmp,timeout=240)
            if proc.returncode:raise RuntimeError(proc.stderr[-2500:])
            result=json.loads(proc.stdout)
    result['credentials_scan_passed']=True
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
