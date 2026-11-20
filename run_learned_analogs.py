"""Disk-backed, chronological radar analog experiment. See outputs/learned_analogs/README.md."""
from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn

from radar.holdout import Scan
from radar.holdout_between import load_tilts
from radar.io import (VOLUME_NAME_RE, distinct_reflectivity_sweep_indices,
                      download_volume, list_archived_volumes, read_volume, volume_name_time)
from radar.grid import grid_reflectivity
from radar.model import DoubleConv
from radar.train import normalize_dbz, denormalize_dbz

ROOT = Path('outputs/learned_analogs')
CACHE = ROOT / 'cache'
EVENTS = {
    '2021-10-21': 'oct21_storm',
    '2021-10-24': 'oct24_storm',
    '2021-10-25': 'oct24_storm',
    '2021-11-05': 'nov06_storm',
    '2021-11-06': 'nov06_storm',
    '2021-11-10': 'nov11_ar',
    '2021-11-11': 'nov11_ar', '2021-11-12': 'nov11_ar',
    '2021-11-15': 'nov15_ar',
    '2021-11-14': 'nov15_ar',
    '2021-11-16': 'nov15_ar',
    '2021-11-18': 'nov18_storm',
    '2021-11-21': 'nov21_storm',
    '2021-11-23': 'nov24_ar',
    '2021-11-24': 'nov24_ar', '2021-11-25': 'nov24_ar',
    '2021-11-27': 'nov27_ar', '2021-11-28': 'nov27_ar',
    '2021-11-29': 'nov27_ar',
}
TRAIN_DAYS = sorted(d for d in EVENTS if d < '2021-11-27')
TEST_DAYS = ['2021-11-27', '2021-11-28']
SHAPE = (320, 320)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def names_for_day(day):
    stamp = day.replace('-', '')
    names = {p.name.removesuffix('.tilts.npz') for p in Path('data/grids').glob(f'KLGX{stamp}_*.tilts.npz')}
    names |= {p.stem for p in CACHE.glob(f'KLGX{stamp}_*.npz') if VOLUME_NAME_RE.fullmatch(p.stem)}
    names |= {p.name for p in Path('data').glob(f'KLGX{stamp}_*') if p.is_file() and VOLUME_NAME_RE.fullmatch(p.name)}
    return sorted(names, key=volume_name_time)


def compact(name):
    path = CACHE / f'{name}.npz'
    if path.exists():
        return path
    day = volume_name_time(name).date().isoformat()
    # Reuse existing all-tilt cache; on new archive volumes grid only the two
    # evaluated tilts, avoiding the cost and memory of all 10-16 sweeps.
    if Path('data/grids',f'{name}.tilts.npz').exists():
        grids = load_tilts(Scan(day, 'stormy', name))[:2]
    else:
        radar=read_volume(download_volume(Scan(day,'stormy',name).uri))
        indices=distinct_reflectivity_sweep_indices(radar)[:2]
        if len(indices)<2: raise ValueError('fewer than two distinct reflectivity tilts')
        grids=[grid_reflectivity(radar,i,missing_gates='invalid') for i in indices]
    if any(g.shape != SHAPE for g in grids):
        raise ValueError(f'{name}: unexpected grid shape')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + '.partial.npz')
    np.savez_compressed(partial, dbz=np.stack([g.dbz for g in grids]).astype('float32'),
                        valid=np.stack([g.valid for g in grids]).astype('uint8'),
                        elevation=np.array([g.sweep_elevation for g in grids]))
    partial.replace(path)
    return path


def collect():
    rows = []
    for day in sorted(EVENTS):
        names = names_for_day(day)
        for name in names:
            try:
                path=compact(name)
                with np.load(path) as z:
                    echo=int(((z['dbz'][0]>=20)&(z['valid'][0]>0)&~blockage()).sum())
                    elevations=[float(v) for v in z['elevation']]
                rows.append(dict(name=name, day=day, event=EVENTS[day], source='unidata-nexrad-level2',
                                 echo20_visible_lowest=echo,elevations=elevations))
            except Exception as exc:
                print('skip', name, str(exc), flush=True)
        print('cached', day, sum(r['day'] == day for r in rows), flush=True)
    rows.sort(key=lambda x: x['name'])
    write_json(ROOT / 'dataset_manifest.json', rows)
    write_json(ROOT / 'splits.json', {'train_days': TRAIN_DAYS, 'test_days': TEST_DAYS,
              'folds': {'nov27': {'train_days':TRAIN_DAYS,'test_days':TEST_DAYS},
                        'nov15': {'train_days':[d for d in EVENTS if d < '2021-11-14'],
                                  'test_days':['2021-11-14','2021-11-15']}},
              'events': EVENTS, 'rule': 'all test days and their event absent from training; analog time < query time, distinct day and event'})
    write_json(ROOT / 'eligible_pairs.json', pairs(rows))
    return rows


def expand(days, max_per_day=12):
    """Incrementally list public archive days and cache pairs; safe to rerun."""
    for day in days:
        all_names = list_archived_volumes('KLGX', date.fromisoformat(day))
        # Keep adjacent scans for causal past context, spaced across the day.
        stamps = [volume_name_time(n) for n in all_names]
        eligible = []
        for i in range(1, len(all_names)):
            age = (stamps[i] - stamps[i-1]).total_seconds() / 60
            if 0 < age <= 12:
                eligible.append(i)
        chosen = [eligible[int(j)] for j in np.linspace(0,len(eligible)-1,min(max_per_day,len(eligible)))] if eligible else []
        selected = [name for i in chosen for name in (all_names[i-1],all_names[i])]
        for name in dict.fromkeys(selected):
            compact(name)
        print('expanded', day, len(set(selected)), flush=True)


def hydrate(manifest_path=ROOT/'dataset_manifest.json'):
    """Rebuild the exact compact collection named in a saved manifest."""
    rows=json.loads(Path(manifest_path).read_text())
    for index,row in enumerate(rows,1):
        compact(row['name'])
        if index%25==0: print('hydrated',index,'of',len(rows),flush=True)


class LRU:
    def __init__(self, size=4):
        self.size, self.items = size, OrderedDict()

    def get(self, name):
        if name not in self.items:
            with np.load(CACHE / f'{name}.npz') as z:
                self.items[name] = (z['dbz'], z['valid'].astype(bool))
        self.items.move_to_end(name)
        while len(self.items) > self.size:
            self.items.popitem(last=False)
        return self.items[name]


def blockage():
    yy, xx = np.indices(SHAPE)
    dy, dx = yy-160, xx-160
    radius = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(dy, dx))
    # A stationary 60-degree azimuth obstruction with an unblocked near range.
    return (radius >= 20) & (radius <= 150) & (angle >= -120) & (angle <= -60)


def pairs(rows):
    by_day = {}
    echo = {r['name']:r.get('echo20_visible_lowest',0) for r in rows}
    elevation = {r['name']:r.get('elevations') for r in rows}
    for r in rows:
        by_day.setdefault(r['day'], []).append(r['name'])
    result=[]
    for day, names in by_day.items():
        names.sort(key=volume_name_time)
        for past, center in zip(names, names[1:]):
            age=(volume_name_time(center)-volume_name_time(past)).total_seconds()/60
            same_tilts=(elevation[center] is not None and elevation[past] is not None and
                        all(abs(a-b)<=0.18 for a,b in zip(elevation[center],elevation[past])))
            if 0 < age <= 12 and echo[center] >= 4000 and same_tilts:
                result.append(dict(center=center, past=past, age=age, day=day, event=EVENTS[day]))
    return sorted(result, key=lambda p:p['center'])


def descriptor(dbz, valid, hidden):
    # Coarse visible pattern. Hidden values are zeroed before reduction.
    field=np.where(valid & ~hidden, normalize_dbz(dbz), 0).reshape(20,16,20,16).mean((1,3))
    flag=(valid & ~hidden).reshape(20,16,20,16).mean((1,3))
    return field, flag


def retrieve(rows, ps):
    cache=LRU(3); hole=blockage(); details=[]
    features={}
    for p in ps:
        cur,cv=cache.get(p['center']); past,pv=cache.get(p['past'])
        for tilt in range(2):
            features[p['center'],tilt]=(descriptor(cur[tilt],cv[tilt],hole),
                                        descriptor(past[tilt],pv[tilt],hole))
    for q in ps:
        for tilt in range(2):
            (qf,qflag),(qpf,qpflag)=features[q['center'],tilt]
            scored=[]
            for a in ps:
                if a['day'] >= q['day'] or a['event']==q['event']:
                    continue
                (af,aflag),(apf,apflag)=features[a['center'],tilt]
                both=(qflag>.5)&(aflag>.5)
                bothp=(qpflag>.5)&(apflag>.5)
                if both.sum()<30 or bothp.sum()<30: continue
                echo=both & ((qf>.08)|(af>.08))
                echop=bothp & ((qpf>.08)|(apf>.08))
                current=np.mean(abs(qf[both]-af[both]))
                previous=np.mean(abs(qpf[bothp]-apf[bothp]))
                if echo.sum()>=5: current=.7*np.mean(abs(qf[echo]-af[echo]))+.3*current
                if echop.sum()>=5: previous=.7*np.mean(abs(qpf[echop]-apf[echop]))+.3*previous
                score=float(current+0.5*previous)
                scored.append((score,a))
            scored.sort(key=lambda item:(item[0],item[1]['center']))
            chosen=scored[:3]
            if any(a['day'] >= q['day'] or a['event'] == q['event'] for _,a in chosen):
                raise AssertionError('chronology or event exclusion violated')
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
            context=np.stack((mean,spread,coverage,confidence))
            np.save(CACHE/f"{q['center']}.tilt{tilt}.analog.npy",context)
            details.append({'query':q['center'],'day':q['day'],'event':q['event'],'tilt':tilt,
                            'matches':[{'center':a['center'],'day':a['day'],'event':a['event'],'score':s} for s,a in chosen]})
        print('retrieved',q['center'],flush=True)
    write_json(ROOT/'retrieval_examples.json',details)


class Net(nn.Module):
    def __init__(self):
        super().__init__(); b=16
        self.e1=DoubleConv(10,b); self.e2=DoubleConv(b,2*b); self.e3=DoubleConv(2*b,4*b)
        self.pool=nn.AvgPool2d(2); self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.d2=DoubleConv(6*b,2*b); self.d1=DoubleConv(3*b,b); self.head=nn.Conv2d(b,1,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b))
        return torch.sigmoid(self.head(self.d1(torch.cat((self.up(self.d2(torch.cat((self.up(c),b),1))),a),1))))


def input_arrays(p,tilt,cache,analog):
    cur,valid=cache.get(p['center']); past,pvalid=cache.get(p['past']); hole=blockage()
    cv=valid[tilt] & ~hole; pv=pvalid[tilt] & ~hole
    x=np.zeros((10,*SHAPE),dtype='float32')
    x[0]=np.where(cv,normalize_dbz(cur[tilt]),0); x[1]=cv
    x[2]=hole; x[3]=np.where(pv,normalize_dbz(past[tilt]),0); x[4]=pv
    if analog: x[5:9]=np.load(CACHE/f"{p['center']}.tilt{tilt}.analog.npy")
    x[9]=tilt
    y=normalize_dbz(cur[tilt]); m=hole & valid[tilt]
    return x,y,m


def train_and_score(ps,steps=250,seeds=(7,19),crop=64,fold='nov27'):
    torch.set_num_threads(4)
    device='cpu'; cache=LRU(5)
    if fold=='nov27':
        train_days,test_days,out=TRAIN_DAYS,TEST_DAYS,ROOT
    elif fold=='nov15':
        test_days=['2021-11-14','2021-11-15']
        train_days=[d for d in EVENTS if d < min(test_days) and EVENTS[d]!='nov15_ar']
        out=ROOT/'fold_nov15'
    elif fold=='nov18':
        test_days=['2021-11-18']
        train_days=[d for d in EVENTS if d < min(test_days) and EVENTS[d]!='nov18_storm']
        out=ROOT/'fold_nov18'
    else: raise ValueError(fold)
    out.mkdir(parents=True,exist_ok=True)
    train=[p for p in ps if p['day'] in train_days]
    test=[p for p in ps if p['day'] in test_days]
    if {p['event'] for p in train} & {p['event'] for p in test}:
        raise AssertionError('training and test events overlap')
    if not train or not test: raise RuntimeError('need training and test pairs')
    scores=[]
    for seed in seeds:
        # Same sample sequence and initialization for both arms.
        for analog in (False,True):
            rng=np.random.default_rng(seed); torch.manual_seed(seed)
            net=Net().to(device); opt=torch.optim.Adam(net.parameters(),lr=8e-4)
            net.train()
            history=[]
            for step in range(steps):
                xs=[];ys=[];ms=[]
                for _ in range(8):
                    p=train[int(rng.integers(len(train)))]; tilt=int(rng.integers(2))
                    x,y,m=input_arrays(p,tilt,cache,analog)
                    rr,cc=np.where(m)
                    if not len(rr): continue
                    j=int(rng.integers(len(rr)))
                    r=int(np.clip(rr[j]-crop//2,0,SHAPE[0]-crop));c=int(np.clip(cc[j]-crop//2,0,SHAPE[1]-crop))
                    xs.append(x[:,r:r+crop,c:c+crop]);ys.append(y[r:r+crop,c:c+crop]);ms.append(m[r:r+crop,c:c+crop])
                if not xs:continue
                xb=torch.from_numpy(np.stack(xs));yb=torch.from_numpy(np.stack(ys)[:,None]);mb=torch.from_numpy(np.stack(ms)[:,None].astype('float32'))
                opt.zero_grad();pred=net(xb)
                weights=mb*(1+3*(yb>=normalize_dbz(np.array(20,dtype='float32'))))
                loss=((pred-yb).abs()*weights).sum()/weights.sum().clamp(min=1)
                loss.backward();opt.step()
                history.append(float(loss.detach()))
                if step%50==0:print('train',seed,analog,step,float(loss.detach()),flush=True)
            label='analog' if analog else 'no_analog'
            torch.save({'model':net.state_dict(),'seed':seed,'steps':steps,'analog':analog,
                        'loss_history':history},out/f'{label}_seed{seed}.pt')
            net.eval()
            with torch.no_grad():
                for p in test:
                    for tilt in range(2):
                        x,y,m=input_arrays(p,tilt,cache,analog)
                        pred=net(torch.from_numpy(x[None])).numpy()[0,0]
                        err=abs(denormalize_dbz(pred)-denormalize_dbz(y))
                        strong=m & (denormalize_dbz(y)>=20)
                        scores.append({'seed':seed,'arm':label,'day':p['day'],'center':p['center'],
                                       'tilt':'lowest' if tilt==0 else 'middle','n':int(m.sum()),
                                       'mae_dbz':float(err[m].mean()) if m.any() else None,
                                       'strong_n':int(strong.sum()),'strong_mae_dbz':float(err[strong].mean()) if strong.any() else None})
            write_json(out/'per_scan_results.json',scores)
    with (out/'per_day_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','arm','day','tilt','n','mae_dbz','strong_n','strong_mae_dbz'])
        writer.writeheader()
        for key in sorted({(r['seed'],r['arm'],r['day'],r['tilt']) for r in scores}):
            subset=[r for r in scores if (r['seed'],r['arm'],r['day'],r['tilt'])==key]
            n=sum(r['n'] for r in subset);sn=sum(r['strong_n'] for r in subset)
            writer.writerow(dict(zip(('seed','arm','day','tilt'),key))|{'n':n,'mae_dbz':sum(r['mae_dbz']*r['n'] for r in subset if r['mae_dbz'] is not None)/n,
                            'strong_n':sn,'strong_mae_dbz':sum(r['strong_mae_dbz']*r['strong_n'] for r in subset if r['strong_mae_dbz'] is not None)/sn if sn else ''})
    with (out/'per_day_results.csv').open() as f:
        day_rows=list(csv.DictReader(f))
    summary={'training_steps_per_arm_seed':steps,'seeds':list(seeds),'train_pairs':len(train),'test_pairs':len(test),'results':{}}
    for tilt in ('lowest','middle'):
        for metric in ('mae_dbz','strong_mae_dbz'):
            rows_here=[r for r in day_rows if r['tilt']==tilt and r[metric]]
            arm_means={arm:float(np.mean([float(r[metric]) for r in rows_here if r['arm']==arm])) for arm in ('no_analog','analog')}
            paired=[]
            for seed in seeds:
                for day in test_days:
                    matches={r['arm']:r for r in rows_here if r['seed']==str(seed) and r['day']==day}
                    if len(matches)==2:
                        paired.append(float(matches['no_analog'][metric])-float(matches['analog'][metric]))
            summary['results'][f'{tilt}_{metric}']={'by_arm':arm_means,'paired_gain_dbz':float(np.mean(paired)) if paired else None}
    summary['fold']=fold
    write_json(out/'summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['collect','retrieve','train','all','expand','hydrate'])
    parser.add_argument('--steps',type=int,default=250);parser.add_argument('--days',nargs='*',default=[])
    parser.add_argument('--max-per-day',type=int,default=12)
    parser.add_argument('--fold',choices=['nov27','nov15','nov18'],default='nov27')
    args=parser.parse_args()
    if args.phase=='expand':expand(args.days,max_per_day=args.max_per_day)
    elif args.phase=='hydrate':hydrate()
    else:
        rows=collect() if args.phase in ('collect','all') else json.loads((ROOT/'dataset_manifest.json').read_text())
        ps=pairs(rows)
        if args.phase in ('retrieve','all'):retrieve(rows,ps)
        if args.phase in ('train','all'):train_and_score(ps,steps=args.steps,fold=args.fold)
