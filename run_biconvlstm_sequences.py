"""Small, reproducible KLGX 15-scan gap-filling experiment."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import run_learned_analogs as legacy
from radar.io import list_archived_volumes, volume_name_time

OUT = Path('outputs/biconvlstm_sequences')
DAYS = ['2021-10-25', '2021-11-06', '2021-11-11', '2021-11-15', '2021-11-18', '2021-11-28']
EVENT = {d: legacy.EVENTS[d] for d in DAYS}
SEEDS = (7, 19)
T = 15
SIZE = 80


def save_json(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n')


def gap_minutes(a,b):
    return (volume_name_time(b)-volume_name_time(a)).total_seconds()/60


def contiguous(names):
    runs=[]; run=[]
    for n in names:
        if run and not 3 <= gap_minutes(run[-1],n) <= 10:
            runs.append(run); run=[]
        run.append(n)
    if run:runs.append(run)
    return runs


def collect():
    cache=legacy.CACHE
    selection_path=OUT/'collection_selection.json'
    selections=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    log=[]
    for day in DAYS:
        if day in selections:
            window=selections[day]
        else:
            names=list_archived_volumes('KLGX',date.fromisoformat(day))
            cached={p.stem for p in cache.glob(f'KLGX{day.replace("-", "")}_*.npz')}
            options=[]
            for run in contiguous(names):
                for i in range(max(0,len(run)-20)):
                    window=run[i:i+21]
                    if len(window)<21:continue
                    present=sum(n in cached for n in window)
                    options.append((present,window))
            if not options:raise RuntimeError(f'No 21-scan continuous archive window on {day}')
            options.sort(key=lambda x:x[0],reverse=True)
            window=options[0][1]
        for i,n in enumerate(window):
            try:legacy.compact(n)
            except Exception as e:log.append({'day':day,'scan':n,'error':str(e)})
            if i%5==0:print('collected',day,i+1,'/',len(window),flush=True)
        selections[day]=window
        save_json('collection_selection.json',selections)
        save_json('collection_errors.json',log)
    manifest()


def manifest():
    selection=json.loads((OUT/'collection_selection.json').read_text())
    sequences=[]; inventory=[]
    for day,names in selection.items():
        available=[n for n in names if (legacy.CACHE/f'{n}.npz').exists()]
        inventory.append({'day':day,'event':EVENT[day],'selected':len(names),'cached':len(available),
                          'start':names[0],'end':names[-1]})
        for run in contiguous(available):
            for i in range(len(run)-T+1):
                ns=run[i:i+T]
                if all(3 <= gap_minutes(a,b) <= 10 for a,b in zip(ns,ns[1:])):
                    sequences.append({'id':f'{day}_{i}_{ns[0]}','day':day,'event':EVENT[day],
                                      'scans':ns,'interval_minutes':[gap_minutes(a,b) for a,b in zip(ns,ns[1:])]})
    save_json('scan_inventory.json',inventory)
    save_json('sequence_manifest.json',sequences)
    folds={}
    training_events={EVENT[d] for d in DAYS[:3]}
    for hold in ('2021-11-15','2021-11-18','2021-11-28'):
        folds[hold]={'train_ids':[s['id'] for s in sequences if s['event'] in training_events],
                     'test_ids':[s['id'] for s in sequences if s['event']==EVENT[hold]],
                     'train_events':sorted(training_events),
                     'test_event':EVENT[hold]}
    save_json('splits.json',{'folds':folds,'seeds':SEEDS,
      'rule':'first three storm events train; final three whole storm events held out, no test event trains any model; no retrieval used'})
    save_json('mask_definitions.json',{'grid_km':4,'shape':[SIZE,SIZE],
      'offline':{'frames_masked':[1,3],'eligible_frame_indices':[1,13],
                 'frame_selection':'deterministic uniform without replacement',
                 'regions':['upper_left','lower_left','upper_right','lower_right','center'],
                 'region_size_pixels':[40,40],'square_size_pixels':[20,30,40],
                 'same_region_and_size_per_sequence':True,'location_resampled_per_masked_frame':True},
      'persistent':{'frame_indices':list(range(T)),
                    'sector':{'center_pixel':[40,40],'radius_km':[20,150],'angle_degrees':[-120,-60]}},
      'deterministic_key':'sum((i+1)*ord(c) for i,c in enumerate(sequence_id)) + seed*1000003 + step*97; step=0 at evaluation',
      'mask_shared_between_models':True,
      'materialized_evaluation_masks':'evaluation_masks.npz'})
    test_ids={i for split in folds.values() for i in split['test_ids']}
    masks={f'{scenario}_seed{seed}_{r["id"]}':mask_for(r,scenario,seed).numpy().astype('uint8')
           for r in sequences if r['id'] in test_ids for scenario in ('offline','persistent') for seed in SEEDS}
    np.savez_compressed(OUT/'evaluation_masks.npz',**masks)
    print('sequences',len(sequences),'by day',dict((d,sum(s['day']==d for s in sequences)) for d in DAYS),flush=True)


def load_scan(name):
    with np.load(legacy.CACHE/f'{name}.npz') as z:
        dbz=z['dbz'].astype('float32'); valid=z['valid'].astype(bool)
    # Valid-area means on 4x4 cells, treating unsampled gates as absent.
    v=valid.reshape(2,SIZE,4,SIZE,4).sum((2,4))
    x=(dbz*valid).reshape(2,SIZE,4,SIZE,4).sum((2,4))/np.maximum(v,1)
    return (np.clip(x,0,60)/60).astype('float32'),(v>=8).astype('float32')


def load_sequences(rows):
    names=sorted({n for r in rows for n in r['scans']})
    cache={n:load_scan(n) for n in names}
    result={}
    for r in rows:
        x=np.stack([cache[n][0] for n in r['scans']],axis=0)
        v=np.stack([cache[n][1] for n in r['scans']],axis=0)
        result[r['id']]=(torch.from_numpy(x),torch.from_numpy(v))
    return result


def sector():
    yy,xx=np.indices((SIZE,SIZE)); dx=xx-40;dy=yy-40
    r=np.hypot(dx,dy)*4; angle=np.degrees(np.arctan2(dy,dx))
    return torch.from_numpy(((r>=20)&(r<=150)&(angle>=-120)&(angle<=-60)).astype('float32'))


def mask_for(row,scenario,seed,step=None):
    # Evaluation masks depend only on sequence and seed, and are shared by arms.
    key=sum((i+1)*ord(c) for i,c in enumerate(row['id'])) + seed*1000003 + (step or 0)*97
    rng=random.Random(key)
    out=torch.zeros(T,1,SIZE,SIZE)
    if scenario=='offline':
        regions=((0,0),(40,0),(0,40),(40,40),(20,20))
        x0,y0=rng.choice(regions);length=rng.choice((20,30,40))
        for t in rng.sample(range(1,T-1),rng.randint(1,3)):
            x=x0+rng.randint(0,40-length);y=y0+rng.randint(0,40-length)
            out[t,0,y:y+length,x:x+length]=1
    else:out[:,0]=sector()
    return out


class ConvGRUCell(nn.Module):
    def __init__(self,ch):
        super().__init__();self.gates=nn.Conv2d(2*ch,2*ch,3,padding=1);self.candidate=nn.Conv2d(2*ch,ch,3,padding=1)
    def forward(self,x,h):
        z,r=self.gates(torch.cat((x,h),1)).sigmoid().chunk(2,1)
        n=torch.tanh(self.candidate(torch.cat((x,r*h),1)))
        return (1-z)*n+z*h


class Net(nn.Module):
    def __init__(self,temporal):
        super().__init__();self.temporal=temporal
        self.enc=nn.Sequential(nn.Conv2d(4,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,8,3,padding=1),nn.ReLU())
        self.down=nn.Sequential(nn.AvgPool2d(4),nn.Conv2d(8,12,3,padding=1),nn.ReLU())
        if temporal:self.forward_cell=ConvGRUCell(12);self.backward_cell=ConvGRUCell(12)
        self.dec=nn.Sequential(nn.Conv2d(32 if temporal else 20,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,2,1))
    def forward(self,x):
        t=x.shape[0];skip=self.enc(x);z=self.down(skip)
        if self.temporal:
            h=torch.zeros_like(z[0:1]); fw=[]
            for i in range(t):h=self.forward_cell(z[i:i+1],h);fw.append(h)
            h=torch.zeros_like(z[0:1]);bw=[None]*t
            for i in range(t-1,-1,-1):h=self.backward_cell(z[i:i+1],h);bw[i]=h
            z=torch.cat((torch.cat(fw),torch.cat(bw)),1)
        z=F.interpolate(z,size=(SIZE,SIZE),mode='bilinear',align_corners=False)
        return torch.sigmoid(self.dec(torch.cat((skip,z),1)))


def infer_input(x,v,m):
    # Original coverage is supplied, but the synthetic missingness mask is not.
    return torch.cat((x*(1-m)*v,v),1)


def score(net,rows,data,scenario,seed,fold,arm):
    records=[];net.eval()
    with torch.no_grad():
        for row in rows:
            x,v=data[row['id']]
            m=mask_for(row,scenario,seed)
            pred=net(infer_input(x,v,m))
            blocked=m.bool() & v.bool()
            for tilt in range(2):
                selected=blocked[:,tilt];strong=selected & (x[:,tilt]>=20/60)
                err=(pred[:,tilt]-x[:,tilt]).abs()*60
                records.append({'scenario':scenario,'fold':fold,'seed':seed,'arm':arm,'sequence_id':row['id'],
                  'day':row['day'],'event':row['event'],'tilt':('lowest','middle')[tilt],
                  'mae_dbz':float(err[selected].mean()) if selected.any() else None,
                  'strong_mae_dbz':float(err[strong].mean()) if strong.any() else None,
                  'blocked_pixels':int(selected.sum()),'strong_pixels':int(strong.sum())})
    return records


def train():
    torch.set_num_threads(4)
    rows=json.loads((OUT/'sequence_manifest.json').read_text())
    splits=json.loads((OUT/'splits.json').read_text())['folds']
    byid={r['id']:r for r in rows}
    records=[]; budget=300
    train_ids=next(iter(splits.values()))['train_ids']
    assert all(s['train_ids']==train_ids for s in splits.values())
    trainrows=[byid[i] for i in train_ids]
    train_data=load_sequences(trainrows)
    for scenario in ('offline','persistent'):
      for seed in SEEDS:
          for arm in ('frame_cnn','bidirectional_gru'):
            torch.manual_seed(seed);rng=random.Random(seed)
            net=Net(arm=='bidirectional_gru');opt=torch.optim.Adam(net.parameters(),lr=.001)
            net.train(); losses=[]
            for step in range(budget):
                row=rng.choice(trainrows);x,v=train_data[row['id']];m=mask_for(row,scenario,seed,step+1)
                pred=net(infer_input(x,v,m))
                hidden=m*v
                weight=hidden*(1+3*(x>=20/60).float())
                loss=((pred-x).abs()*weight).sum()/weight.sum().clamp_min(1)
                opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
            path=OUT/'checkpoints'/scenario/f'{arm}_seed{seed}.pt'
            path.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'model':net.state_dict(),'seed':seed,'steps':budget,'scenario':scenario,
                        'arm':arm,'train_ids':train_ids,'loss_last20':sum(losses[-20:])/20},path)
            print('trained',scenario,seed,arm,'loss',round(sum(losses[-20:])/20,4),flush=True)
            for fold,split in splits.items():
                testrows=[byid[i] for i in split['test_ids']]
                if not testrows:raise RuntimeError(f'empty fold {fold}')
                assert not ({r['event'] for r in trainrows}&{r['event'] for r in testrows})
                # Held-out scans are opened only after this model is trained.
                test_data=load_sequences(testrows)
                records.extend(score(net,testrows,test_data,scenario,seed,fold,arm))
                del test_data
            write_results(records)
    save_json('training_budget.json',{'steps_per_run':budget,'optimizer':'Adam','learning_rate':.001,
      'batch_size_sequences':1,'loss':'blocked valid L1, 4x for target >=20 dBZ','seeds':SEEDS,
      'mask_rule':'same deterministic masks for both arms; 1-3 interior frames offline, all 15 persistent'})


def write_results(records):
    OUT.mkdir(parents=True,exist_ok=True)
    if not records:return
    with (OUT/'per_sequence_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    groups=defaultdict(list)
    for r in records:groups[(r['scenario'],r['fold'],r['seed'],r['arm'],r['day'],r['tilt'])].append(r)
    summary=[]
    for key,items in sorted(groups.items()):
        row=dict(zip(('scenario','fold','seed','arm','day','tilt'),key))
        for field,count in (('mae_dbz','blocked_pixels'),('strong_mae_dbz','strong_pixels')):
            usable=[r for r in items if r[field] is not None]
            row[field]=sum(r[field]*r[count] for r in usable)/sum(r[count] for r in usable) if usable else None
            row[count]=sum(r[count] for r in usable)
        row['sequences']=len(items);summary.append(row)
    with (OUT/'per_day_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['collect','manifest','train']);a=p.parse_args()
    {'collect':collect,'manifest':manifest,'train':train}[a.action]()
