"""Personal portfolio prototype developed with AI assistance.
Official Depth Anything V2 supplies the pretrained models. No training is performed.
This is a one-room, correlated-view case study; it is not a navigation benchmark.
"""
from pathlib import Path
import argparse, gc, hashlib, json, platform, subprocess, sys, time
from collections import Counter
import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO = Path('/content/Depth-Anything-V2')
MODEL_DIR = REPO / 'metric_depth'
DATA = Path('/content/comp1818_data/nyu_depth_v2_labeled.mat')
OUT = Path('/content/comp1818_living_room')
CONFIG = dict(input_size=518, reference_near_metres=2.0,
              depth_range_metres=[0.1,10.0], crop_yx=[45,471,41,601],
              roi_fraction_xy=[0.2,0.8,0.2,0.7], median_kernel=3,
              minimum_component_pixels=32, decision_fraction=0.03,
              calibration_thresholds_metres=[1.5,2.0,2.5], timing_repeats=3,
              brightness_factor=0.6, blur_kernel=7, seed=1818)
MODEL_CONFIGS = {
 'Small':dict(encoder='vits', features=64, out_channels=[48,96,192,384]),
 'Base':dict(encoder='vitb', features=128, out_channels=[96,192,384,768])}


def decode_ref(f, ref):
    return ''.join(chr(int(x)) for x in np.asarray(f[ref]).reshape(-1))


def safe_ratio(a,b):
    return float(a/b) if b else None


def binary_metrics(truth, pred, valid):
    t=np.asarray(truth,bool)[valid]; p=np.asarray(pred,bool)[valid]
    tp=int(np.sum(t&p)); fp=int(np.sum(~t&p)); fn=int(np.sum(t&~p)); tn=int(np.sum(~t&~p))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=safe_ratio(tp,tp+fp),
                recall=safe_ratio(tp,tp+fn),fpr=safe_ratio(fp,fp+tn),
                f1=safe_ratio(2*tp,2*tp+fp+fn))


def depth_metrics(pred, gt, valid):
    if not np.isfinite(pred).all():
        raise ValueError('Non-finite model output; do not silently discard predictions.')
    p=np.asarray(pred,dtype=np.float64)[valid]; g=np.asarray(gt,dtype=np.float64)[valid]
    if not len(g): raise ValueError('No valid measured depth pixels.')
    e=p-g
    return dict(mae=float(np.mean(np.abs(e))),rmse=float(np.sqrt(np.mean(e*e))),
                absrel=float(np.mean(np.abs(e)/g)),valid_depth_pixels=len(g))


def component_filter(mask,min_area):
    count,lab,stats,_=cv2.connectedComponentsWithStats(np.asarray(mask,np.uint8), connectivity=8)
    keep=np.zeros(count,bool)
    if count>1: keep[1:]=stats[1:,cv2.CC_STAT_AREA]>=min_area
    return keep[lab]


def roi_mask(shape):
    h,w=shape; x0,x1,y0,y1=CONFIG['roi_fraction_xy']
    r=np.zeros((h,w),bool);r[int(y0*h):int(y1*h),int(x0*w):int(x1*w)]=True
    return r


def detect(depth,method,threshold,rgb=None):
    """Prediction uses RGB/predicted depth only, never reference labels or depth."""
    r=roi_mask(depth.shape)
    if method=='always_obstacle': return r
    if method=='image_rule':
        edges=cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),50,150)
        edges=cv2.dilate(edges,np.ones((7,7),np.uint8))>0
        return component_filter(edges&r,CONFIG['minimum_component_pixels'])
    d=depth.astype(np.float32)
    if method in ('median','median_regions'): d=cv2.medianBlur(d,CONFIG['median_kernel'])
    m=(d<threshold)&r
    if method=='median_regions': m=component_filter(m,CONFIG['minimum_component_pixels'])
    return m


def measured_mask(gt):
    mask=np.isfinite(gt)&(gt>=CONFIG['depth_range_metres'][0])&(gt<=CONFIG['depth_range_metres'][1])
    crop=np.zeros(gt.shape,bool);y0,y1,x0,x1=CONFIG['crop_yx'];crop[y0:y1,x0:x1]=True
    return mask&crop


def load_view(f,index,floor_ids):
    rgb=np.asarray(f['images'][index]).transpose(2,1,0).copy()
    gt=np.asarray(f['rawDepths'][index]).T.copy() # measured reference, no inpainting
    labels=np.asarray(f['labels'][index]).T.copy()
    valid=measured_mask(gt); obs_valid=valid&roi_mask(gt.shape)&(labels>0)
    truth=(gt<CONFIG['reference_near_metres'])&~np.isin(labels,floor_ids)
    return dict(index=int(index),rgb=rgb,gt=gt,valid=valid,obs_valid=obs_valid,truth=truth,labels=labels)


def select_room(f):
    refs=np.asarray(f['scenes']).reshape(-1)
    scenes=[decode_ref(f,r) for r in refs]
    counts=Counter(s for s in scenes if s.lower().startswith('living_room'))
    if not counts: raise ValueError('No living-room scenes found.')
    # Selection depends only on view count, not on model accuracy or label prevalence.
    room=sorted(counts,key=lambda s:(-counts[s],s))[0]
    ids=[i for i,s in enumerate(scenes) if s==room]
    if len(ids)<6: raise ValueError('Too few views for this within-room split.')
    ndev=max(2,int(len(ids)*.4)); dev=ids[:ndev];gap=ids[ndev:ndev+1];test=ids[ndev+1:]
    if len(test)<2: raise ValueError('Too few reserved test views.')
    names=[decode_ref(f,r) for r in np.asarray(f['names']).reshape(-1)]
    floor_ids=[i+1 for i,n in enumerate(names) if n.strip().lower()=='floor']
    if not floor_ids: raise ValueError('Cannot identify floor class; do not invent a class ID.')
    return dict(room=room,room_counts=dict(sorted(counts.items())),all_indices=ids,
                development_indices=dev,gap_indices=gap,test_indices=test,
                floor_label_ids=floor_ids,selection='largest labelled living-room view count, alphabetical tie break',
                split='ordered first 40 percent development, one view excluded as gap, remainder test; correlated views remain')


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def infer(model,rgb):
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    sync();t=time.perf_counter()
    with torch.inference_mode(): d=model.infer_image(bgr,CONFIG['input_size'])
    sync();ms=(time.perf_counter()-t)*1000
    if d.shape!=rgb.shape[:2] or not np.isfinite(d).all(): raise ValueError('Invalid depth output.')
    return d,ms


def load_model(name,device):
    if str(MODEL_DIR) not in sys.path: sys.path.insert(0,str(MODEL_DIR))
    from depth_anything_v2.dpt import DepthAnythingV2
    cfg=MODEL_CONFIGS[name]; ckpt=MODEL_DIR/'checkpoints'/f"depth_anything_v2_metric_hypersim_{cfg['encoder']}.pth"
    if not ckpt.is_file(): raise FileNotFoundError(ckpt)
    model=DepthAnythingV2(**cfg,max_depth=20)
    model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True))
    return model.to(device).eval(),ckpt


def pooled_metrics(rows):
    counts={k:int(sum(r[k] for r in rows)) for k in ['tp','fp','fn','tn']}
    a=counts
    return dict(**counts,precision=safe_ratio(a['tp'],a['tp']+a['fp']),recall=safe_ratio(a['tp'],a['tp']+a['fn']),
                fpr=safe_ratio(a['fp'],a['fp']+a['tn']),f1=safe_ratio(2*a['tp'],2*a['tp']+a['fp']+a['fn']))


def score(view,depth,model,condition,threshold,methods):
    rows=[];r=roi_mask(depth.shape)
    for method in methods:
        t=time.perf_counter(); pred=detect(depth,method,threshold,view['rgb']); detector_ms=(time.perf_counter()-t)*1000
        metrics=binary_metrics(view['truth'],pred,view['obs_valid'])
        # Decisions use the entire ROI, including pixels with no reference measurement.
        pred_decision=bool(pred[r].mean()>=CONFIG['decision_fraction'])
        # Reference decision is defined on labelled, measured ROI pixels only.
        ref_decision=bool(view['truth'][view['obs_valid']].mean()>=CONFIG['decision_fraction'])
        rows.append(dict(index=view['index'],model=model,condition=condition,method=method,
                         detector_ms=detector_ms,pred_stop=int(pred_decision),reference_stop=int(ref_decision),
                         labelled_roi_pixels=int(view['obs_valid'].sum()),**metrics))
    return rows


def save_visual(view,pred,threshold):
    fig,axes=plt.subplots(1,5,figsize=(18,4))
    axes[0].imshow(view['rgb']);axes[0].set_title(f"RGB view {view['index']}")
    axes[1].imshow(np.where(view['valid'],view['gt'],np.nan),vmin=0,vmax=10,cmap='viridis');axes[1].set_title('Measured depth 0 to 10 m')
    axes[2].imshow(pred,vmin=0,vmax=10,cmap='viridis');axes[2].set_title('Predicted depth 0 to 10 m')
    reference_cmap=plt.get_cmap('gray').copy();reference_cmap.set_bad('#aab3bc')
    axes[3].imshow(np.where(view['obs_valid'],view['truth'],np.nan),vmin=0,vmax=1,cmap=reference_cmap);axes[3].set_title('Reference mask\nWhite near nonfloor; grey unscored')
    axes[4].imshow(view['rgb']);mask=detect(pred,'median_regions',threshold)
    overlay=np.zeros((*mask.shape,4));overlay[mask]=[1,0,0,.55];axes[4].imshow(overlay);axes[4].set_title('Filtered prediction overlay')
    for ax in axes:ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'figures'/f"qualitative_{view['index']}.png",dpi=150);plt.close(fig)


def replot_saved():
    meta=json.loads((OUT/'run_metadata.json').read_text())
    selection=meta['selection']
    with h5py.File(DATA,'r') as f:
        for index in selection['test_indices'][:3]:
            view=load_view(f,index,selection['floor_label_ids'])
            pred=np.load(OUT/'predictions'/f'Small_clean_{index}.npy')
            save_visual(view,pred,meta['prediction_threshold_metres'])
    print('Regenerated qualitative figures from saved predictions; no model rerun.')


def make_plots(depth_df,obs_summary,timing):
    clean=depth_df[depth_df.condition=='clean']
    means=clean.groupby('model')[['mae','rmse','absrel']].mean()
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    for ax,col,title in zip(axes,['mae','rmse','absrel'],['MAE in metres','RMSE in metres','Absolute relative error']):
        means[col].plot.bar(ax=ax,rot=0,color=['#2878A0','#E49A36'][:len(means)]);ax.set_title(title);ax.set_xlabel('')
    fig.suptitle('One living room  Mean of per-view depth metrics');fig.tight_layout();fig.savefig(OUT/'figures'/'depth_accuracy.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4));s=obs_summary[obs_summary.condition=='clean'].copy();s['label']=s.model+' '+s.method
    s.set_index('label')[['precision','recall','fpr']].plot.bar(ax=ax);ax.set_ylim(0,1.05);ax.set_ylabel('Ratio');ax.set_xlabel('');fig.tight_layout();fig.savefig(OUT/'figures'/'obstacle_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.5));pd.DataFrame(timing).set_index('model')[['median_ms','p95_ms']].plot.bar(ax=ax,rot=0);ax.set_ylabel('Inference milliseconds per image');ax.set_xlabel('');fig.tight_layout();fig.savefig(OUT/'figures'/'inference_latency.png',dpi=160);plt.close(fig)
    robust=depth_df[depth_df.model=='Small'].groupby('condition')[['mae','rmse','absrel']].mean()
    robust.to_csv(OUT/'robustness_depth_summary.csv')


def run(models=('Small','Base')):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True);(OUT/'predictions').mkdir(exist_ok=True)
    np.random.seed(CONFIG['seed']);torch.manual_seed(CONFIG['seed'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    env=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__,
             device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
             source_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
             ai_assistance='Developed with OpenAI Codex assistance; pretrained models credited to Depth Anything V2 authors.',
             config=CONFIG)
    with h5py.File(DATA,'r') as f:
        selection=select_room(f);(OUT/'selection.json').write_text(json.dumps(selection,indent=2))
        print('Selected:',selection['room'],'development:',selection['development_indices'],'gap:',selection['gap_indices'],'test:',selection['test_indices'],flush=True)
        views={i:load_view(f,i,selection['floor_label_ids']) for i in selection['all_indices']}
    test=[views[i] for i in selection['test_indices']];dev=[views[i] for i in selection['development_indices']]
    for v in test+dev:
        if not v['obs_valid'].any():raise ValueError('No labelled measured pixels in evaluation ROI.')
    depth_rows=[];obs_rows=[];timing=[];calibration=[];threshold=2.0
    for name in models:
        print('Loading',name,flush=True);model,ckpt=load_model(name,device)
        env[name+'_checkpoint_sha256']=hashlib.sha256(ckpt.read_bytes()).hexdigest()
        for _ in range(2):infer(model,dev[0]['rgb'])
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        if name=='Small':
            dev_pred={v['index']:infer(model,v['rgb'])[0] for v in dev}
            for th in CONFIG['calibration_thresholds_metres']:
                rs=[binary_metrics(v['truth'],detect(dev_pred[v['index']],'raw',th),v['obs_valid']) for v in dev]
                calibration.append(dict(threshold=th,**pooled_metrics(rs)))
            threshold=sorted(calibration,key=lambda r:(-(r['f1'] if r['f1'] is not None else -1),abs(r['threshold']-2.0),r['threshold']))[0]['threshold']
            print('Development-only selected prediction threshold:',threshold,flush=True)
        latencies=[]
        for n,v in enumerate(test):
            pred,ms=infer(model,v['rgb']);repeat=[ms]
            for _ in range(CONFIG['timing_repeats']-1):repeat.append(infer(model,v['rgb'])[1])
            latencies.extend(repeat)
            np.save(OUT/'predictions'/f"{name}_clean_{v['index']}.npy",pred)
            depth_rows.append(dict(index=v['index'],model=name,condition='clean',**depth_metrics(pred,v['gt'],v['valid'])))
            methods=['raw','median','median_regions']
            if name=='Small':methods+=['always_obstacle','image_rule']
            obs_rows.extend(score(v,pred,name,'clean',threshold,methods))
            if name=='Small' and n<3:save_visual(v,pred,threshold)
            if name=='Small':
                for cond,img in [('brightness_0.6',np.clip(v['rgb'].astype(float)*.6,0,255).astype(np.uint8)),('blur_7',cv2.GaussianBlur(v['rgb'],(7,7),0))]:
                    d,_=infer(model,img);np.save(OUT/'predictions'/f"{name}_{cond}_{v['index']}.npy",d)
                    depth_rows.append(dict(index=v['index'],model=name,condition=cond,**depth_metrics(d,v['gt'],v['valid'])))
                    # Image-rule baseline is not used here; the depth-based extension uses only d.
                    obs_rows.extend(score(v,d,name,cond,threshold,['raw','median','median_regions']))
            print(name,'view',n+1,'/',len(test),'index',v['index'],flush=True)
        timing.append(dict(model=name,median_ms=float(np.median(latencies)),p95_ms=float(np.percentile(latencies,95)),
                           repeats_per_view=CONFIG['timing_repeats'],test_views=len(test),
                           peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None))
        del model;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    depth_df=pd.DataFrame(depth_rows);obs_df=pd.DataFrame(obs_rows)
    depth_df.to_csv(OUT/'depth_per_view.csv',index=False);obs_df.to_csv(OUT/'obstacles_per_view.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'development_thresholds.csv',index=False)
    pd.DataFrame(timing).to_csv(OUT/'timing.csv',index=False)
    summary=[]
    for key,g in obs_df.groupby(['model','condition','method']):
        rs=g.to_dict('records');p=pooled_metrics(rs)
        decision=binary_metrics(g.reference_stop.to_numpy(bool),g.pred_stop.to_numpy(bool),np.ones(len(g),bool))
        summary.append(dict(model=key[0],condition=key[1],method=key[2],views=len(g),
                            mean_detector_ms=float(g.detector_ms.mean()),decision_recall=decision['recall'],
                            decision_fpr=decision['fpr'],**p))
    obs_summary=pd.DataFrame(summary);obs_summary.to_csv(OUT/'obstacle_summary.csv',index=False)
    depth_df.groupby(['model','condition'])[['mae','rmse','absrel']].mean().to_csv(OUT/'depth_summary.csv')
    env['prediction_threshold_metres']=threshold;env['selection']=selection
    (OUT/'run_metadata.json').write_text(json.dumps(env,indent=2))
    make_plots(depth_df,obs_summary,timing)
    (OUT/'INTERPRETATION_LIMITS.txt').write_text('One NYU living-room scene only. Views are correlated, even with a one-view gap. No training or fine-tuning occurred. Selection used frame count only, not accuracy. Depth uses measured rawDepths, valid range 0.1 to 10 m, crop rows [45,471) and columns [41,601). No test-time depth scale alignment. Obstacles are near nonfloor reference pixels in a predefined ROI, not physical collision labels. Floor class labels are used only to construct the scoring reference; predictions may flag floors and are penalised for this. Image edge baseline is a simple heuristic, not evidence of a formal symbolic reasoning system. Per-view decision reference uses measured labelled pixels while prediction uses full ROI; missing reference pixels limit interpretation. Micro-pooled pixel precision/recall/FPR and mean per-view depth metrics are distinct aggregates. Undefined ratios are blank, not zero. Small-only brightness and blur tests are synthetic perturbations. Inference timing includes image preprocessing and GPU synchronisation but excludes model loading, disk I/O and detector steps. No robot movement or safety validity. Personal portfolio prototype developed with AI assistance.')
    print('COMPLETE',OUT,flush=True)
    print(depth_df.groupby(['model','condition'])[['mae','rmse','absrel']].mean().to_string(),flush=True)
    print(obs_summary[['model','condition','method','precision','recall','fpr','f1']].to_string(index=False),flush=True)
    return OUT


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',default=['Small','Base'],choices=list(MODEL_CONFIGS))
    parser.add_argument('--repo',type=Path,default=REPO,help='Official Depth Anything V2 checkout')
    parser.add_argument('--data',type=Path,default=DATA,help='NYU labelled MAT file')
    parser.add_argument('--out',type=Path,default=OUT,help='Output directory')
    args=parser.parse_args()
    REPO=args.repo.resolve();MODEL_DIR=REPO/'metric_depth';DATA=args.data.resolve();OUT=args.out.resolve()
    run(args.models)
