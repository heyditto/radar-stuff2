"""Expanded, intensity-aware historical-analog ablation.

Run after expanding the shared compact scan cache with run_learned_analogs.py.
Outputs are isolated in outputs/learned_analogs/v2.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

import run_learned_analogs as base
from radar.io import volume_name_time
from radar.train import normalize_dbz, denormalize_dbz
from radar.model import DoubleConv

ROOT = Path('outputs/learned_analogs/v2')
SHAPE = base.SHAPE


@lru_cache(maxsize=None)
def sounding_for(name):
    """Latest already-observed UIL surface sounding within 24 hours."""
    at=volume_name_time(name)
    available=[]
    for path in Path('data/soundings').glob('UIL_*.json'):
        stamp=datetime.fromisoformat(path.stem.split('_')[1].replace('Z','+00:00'))
        if stamp <= at-timedelta(hours=1) and at-stamp <= timedelta(hours=24):
            available.append((stamp,path))
    if not available:
        return np.zeros(4,dtype='float32'),None
    stamp,path=max(available)
    profile=json.loads(path.read_text())['profiles'][0]['profile']
    surface=next((r for r in profile if all(r.get(k) is not None for k in ('tmpc','dwpc','sknt'))),None)
    if surface is None:
        return np.zeros(4,dtype='float32'),None
    t=float(surface['tmpc']); dew=float(surface['dwpc']); wind=float(surface['sknt'])
    return np.array([np.clip(t/30,-1,1),np.clip((t-dew)/30,0,1),
                     np.clip(wind/50,0,1),1],dtype='float32'),str(path)


def collect():
    rows = []
    hole = base.blockage()
    for day in sorted(base.EVENTS):
        for name in base.names_for_day(day):
            path = base.CACHE / f'{name}.npz'
            if not path.exists():
                continue
            with np.load(path) as z:
                echo = int(((z['dbz'][0] >= 20) & (z['valid'][0] > 0) & ~hole).sum())
                elevations = [float(v) for v in z['elevation']]
            rows.append(dict(name=name, day=day, event=base.EVENTS[day], source='unidata-nexrad-level2',
                             echo20_visible_lowest=echo, elevations=elevations))
    rows.sort(key=lambda r: r['name'])
    ps = base.pairs(rows)
    base.write_json(ROOT/'dataset_manifest.json', rows)
    base.write_json(ROOT/'eligible_pairs.json', ps)
    base.write_json(ROOT/'weather_manifest.json',[
        {'center':p['center'],'source':sounding_for(p['center'])[1],
         'features':sounding_for(p['center'])[0].tolist()} for p in ps])
    folds = {}
    for fold, days in {'nov15':['2021-11-14','2021-11-15'],
                       'nov18':['2021-11-18'],
                       'nov27':['2021-11-27','2021-11-28']}.items():
        event = base.EVENTS[days[0]]
        train_days = [d for d in sorted(base.EVENTS) if d < min(days) and base.EVENTS[d] != event]
        folds[fold] = dict(train_days=train_days, test_days=days,
                           train_centers=[p['center'] for p in ps if p['day'] in train_days],
                           test_centers=[p['center'] for p in ps if p['day'] in days])
    base.write_json(ROOT/'splits.json', {'events':base.EVENTS, 'folds':folds,
        'rule':'whole UTC days and storm events held out; candidates precede query day and have a different event'})
    print('collected',len(rows),'scans',len(ps),'eligible centers',flush=True)
    return ps


def desc(dbz, valid, hole):
    visible = valid & ~hole
    f, flag = base.descriptor(dbz, valid, hole)
    vals = dbz[visible]
    rain = vals >= 20
    stats = np.array([rain.mean() if len(vals) else 0,
                      np.mean(vals >= 35) if len(vals) else 0,
                      np.percentile(vals, 98) / 60 if len(vals) else 0], dtype='float32')
    return f, flag, stats


def retrieve(ps):
    cache=base.LRU(4); hole=base.blockage(); features={}; details=[]
    for p in ps:
        cur,cv=cache.get(p['center']); past,pv=cache.get(p['past'])
        for tilt in range(2):
            features[p['center'],tilt]=(desc(cur[tilt],cv[tilt],hole),desc(past[tilt],pv[tilt],hole))
    for qi,q in enumerate(ps):
        qw,_=sounding_for(q['center'])
        for tilt in range(2):
            (qf,qflag,qs),(qpf,qpflag,qps)=features[q['center'],tilt]
            scored=[]
            for a in ps:
                if a['day']>=q['day'] or a['event']==q['event']: continue
                (af,aflag,ast),(apf,apflag,apst)=features[a['center'],tilt]
                both=(qflag>.5)&(aflag>.5); bothp=(qpflag>.5)&(apflag>.5)
                if both.sum()<30 or bothp.sum()<30: continue
                active=both & ((qf>.08)|(af>.08)); activep=bothp & ((qpf>.08)|(apf>.08))
                current=float(np.mean(abs(qf[both]-af[both])))
                previous=float(np.mean(abs(qpf[bothp]-apf[bothp])))
                if active.sum()>=5: current=.7*float(np.mean(abs(qf[active]-af[active])))+.3*current
                if activep.sum()>=5: previous=.7*float(np.mean(abs(qpf[activep]-apf[activep])))+.3*previous
                # Visible rain extent and upper intensity distinguish lookalike
                # coarse fields with very different rain rates.
                intensity=float(np.abs(qs-ast)@[2.0,2.0,.4]+.5*(np.abs(qps-apst)@[2.0,2.0,.4]))
                aw,_=sounding_for(a['center'])
                weather=.2*float(np.abs(qw[:3]-aw[:3]).mean()) if qw[3] and aw[3] else 0
                scored.append((current+.5*previous+intensity+weather,a))
            scored.sort(key=lambda item:(item[0],item[1]['center']))
            chosen=scored[:3]
            if chosen:
                fields=[]; flags=[]
                for _,a in chosen:
                    dbz,valid=cache.get(a['center'])
                    fields.append(normalize_dbz(dbz[tilt])); flags.append(valid[tilt])
                f=np.stack(fields); v=np.stack(flags)
                count=v.sum(0)
                mean=np.where(count>0,(f*v).sum(0)/np.maximum(count,1),0).astype('float32')
                spread=np.where(count>1,np.sqrt(((f-mean)**2*v).sum(0)/np.maximum(count,1)),0).astype('float32')
                coverage=(count/len(chosen)).astype('float32')
                confidence=np.full(SHAPE,1/(1+chosen[0][0]*10),dtype='float32')
            else:
                mean=spread=coverage=confidence=np.zeros(SHAPE,dtype='float32')
            np.save(base.CACHE/f"{q['center']}.tilt{tilt}.analog_v2.npy",np.stack((mean,spread,coverage,confidence)))
            details.append({'query':q['center'],'day':q['day'],'event':q['event'],'tilt':tilt,
                'visible_rain_fraction':float(qs[0]),
                'matches':[{'center':a['center'],'day':a['day'],'event':a['event'],'score':float(s)} for s,a in chosen]})
        if qi%25==0:print('retrieved',qi+1,'/',len(ps),flush=True)
    base.write_json(ROOT/'retrieval_examples.json',details)


class GatedNet(base.Net):
    def __init__(self):
        super().__init__()
        self.e1=DoubleConv(14,16)
        self.gate=nn.Conv2d(10,1,1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias,-2.0)

    def gate_strength(self,x):
        visible=x[:,[0,1,3,4,5,6,7,8]]
        current_rain=((x[:,0:1]>=20/60)*x[:,1:2]).float().mean((2,3),keepdim=True)
        past_rain=((x[:,3:4]>=20/60)*x[:,4:5]).float().mean((2,3),keepdim=True)
        signal=torch.cat((visible,current_rain.expand_as(x[:,0:1]),
                          past_rain.expand_as(x[:,0:1])),dim=1)
        return torch.sigmoid(self.gate(signal))

    def forward(self,x):
        strength=self.gate_strength(x)
        x=torch.cat((x[:,:5],x[:,5:9]*strength,x[:,9:]),dim=1)
        return super().forward(x)


def input_arrays(p,tilt,cache,analog):
    x,y,m=base_input_arrays(p,tilt,cache,False)
    if analog:
        x[5:9]=np.load(base.CACHE/f"{p['center']}.tilt{tilt}.analog_v2.npy")
    weather,_=sounding_for(p['center'])
    x=np.concatenate((x,np.broadcast_to(weather[:,None,None],(4,*SHAPE)).copy()),axis=0)
    return x,y,m


base_input_arrays=base.input_arrays


def train(ps,steps,folds,seeds):
    base.ROOT=ROOT
    base.Net=GatedNet
    base.input_arrays=input_arrays
    for fold in folds:
        split=json.loads((ROOT/'splits.json').read_text())['folds'][fold]
        base.TRAIN_DAYS=split['train_days']
        base.TEST_DAYS=split['test_days']
        base.train_and_score(ps,steps=steps,seeds=seeds,fold=fold)


def validate(ps):
    splits=json.loads((ROOT/'splits.json').read_text())['folds']
    details=json.loads((ROOT/'retrieval_examples.json').read_text())
    by_center={p['center']:p for p in ps}
    assert len(by_center)==len(ps)
    assert len(details)==2*len(ps)
    for fold,split in splits.items():
        train=[by_center[n] for n in split['train_centers']]
        test=[by_center[n] for n in split['test_centers']]
        assert train and test,fold
        assert not ({p['event'] for p in train}&{p['event'] for p in test}),fold
        assert max(p['day'] for p in train)<min(p['day'] for p in test),fold
    for r in details:
        q=by_center[r['query']]
        for m in r['matches']:
            a=by_center[m['center']]
            assert a['day']<q['day'] and a['event']!=q['event']
    for p in ps:
        _,source=sounding_for(p['center'])
        if source:
            stamp=datetime.fromisoformat(Path(source).stem.split('_')[1].replace('Z','+00:00'))
            assert stamp+timedelta(hours=1)<=volume_name_time(p['center'])
    print('validated',len(ps),'centers',len(details),'query/tilt retrievals',flush=True)


def diagnose(ps,folds,seeds):
    torch.set_num_threads(4)
    by_center={p['center']:p for p in ps}
    splits=json.loads((ROOT/'splits.json').read_text())['folds']
    cache=base.LRU(5)
    for fold in folds:
        out=ROOT if fold=='nov27' else ROOT/f'fold_{fold}'
        records=[]
        for seed in seeds:
            net=GatedNet()
            checkpoint=torch.load(out/f'analog_seed{seed}.pt',map_location='cpu',weights_only=False)
            net.load_state_dict(checkpoint['model']);net.eval()
            with torch.no_grad():
                for center in splits[fold]['test_centers']:
                    p=by_center[center]
                    for tilt in range(2):
                        x,y,m=input_arrays(p,tilt,cache,True)
                        xb=torch.from_numpy(x[None])
                        gate=net.gate_strength(xb).numpy()[0,0]
                        ablated=xb.clone();ablated[:,5:9]=0
                        pred=net(ablated).numpy()[0,0]
                        err=abs(denormalize_dbz(pred)-denormalize_dbz(y))
                        strong=m & (denormalize_dbz(y)>=20)
                        records.append({'seed':seed,'center':center,'day':p['day'],'tilt':tilt,
                            'gate_hidden_mean':float(gate[m].mean()) if m.any() else None,
                            'ablated_mae_dbz':float(err[m].mean()) if m.any() else None,
                            'ablated_strong_mae_dbz':float(err[strong].mean()) if strong.any() else None,
                            'n':int(m.sum()),'strong_n':int(strong.sum())})
        base.write_json(out/'gate_diagnostics.json',records)
        print('diagnosed',fold,len(records),'scan/tilt/seed records',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['collect','retrieve','train','validate','diagnose','all'])
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--folds',nargs='+',default=['nov15','nov18','nov27'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[7,19])
    args=parser.parse_args()
    ps=collect() if args.phase in ('collect','all') else json.loads((ROOT/'eligible_pairs.json').read_text())
    if args.phase in ('retrieve','all'):retrieve(ps)
    if args.phase in ('validate','all'):validate(ps)
    if args.phase in ('train','all'):train(ps,args.steps,args.folds,args.seeds)
    if args.phase in ('diagnose','all'):diagnose(ps,args.folds,args.seeds)
