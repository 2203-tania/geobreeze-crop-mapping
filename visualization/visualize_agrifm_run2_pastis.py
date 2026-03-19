"""Visualization script for AgriFM PASTIS Run 2 - uses real model predictions."""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from pathlib import Path
import h5py
import warnings
warnings.filterwarnings('ignore')

_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})

sys.path.insert(0, '/mnt/tania')
import mmcv_mock_module

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BASE_OUTPUT_DIR = Path('/mnt/tania/geobreeze/outputs')
RUN2_DIR        = Path('/mnt/tania/agrifm_pastis_run_v2')
CHECKPOINT_PATH = RUN2_DIR / 'best_mIoU_epoch_25.pth'
DATA_DIR        = Path('/mnt/tania/pastis_agrifm/h5_samples')
TEST_LIST       = Path('/mnt/tania/pastis_agrifm/data_list/test.txt')
AGRIFM_REPO     = '/mnt/tania/AgriFM'

PASTIS_CLASSES = {
    0:  ('Background',        '#000000'),
    1:  ('Meadow',            '#7CFC00'),
    2:  ('Soft Winter Wheat', '#FFD700'),
    3:  ('Corn',              '#FF8C00'),
    4:  ('Winter Barley',     '#DAA520'),
    5:  ('Winter Rapeseed',   '#ADFF2F'),
    6:  ('Spring Barley',     '#F0E68C'),
    7:  ('Sunflower',         '#FFA500'),
    8:  ('Grapevine',         '#8B0000'),
    9:  ('Beet',              '#FF69B4'),
    10: ('Winter Triticale',  '#BDB76B'),
    11: ('Winter Durum Wht',  '#EEE8AA'),
    12: ('Fruits & Veg',      '#228B22'),
    13: ('Potatoes',          '#D2691E'),
    14: ('Legum. Fodder',     '#90EE90'),
    15: ('Soybeans',          '#556B2F'),
    16: ('Orchard',           '#006400'),
    17: ('Mixed Cereals',     '#F5DEB3'),
    18: ('Sorghum',           '#CD853F'),
    19: ('Void Label',        '#808080'),
    20: ('Unknown',           '#C0C0C0'),
}
NUM_CLASSES = 21
CMAP_COLORS = [PASTIS_CLASSES[i][1] for i in range(NUM_CLASSES)]
PASTIS_CMAP = ListedColormap(CMAP_COLORS)
BG = '#0d0d0d'
lkw = dict(cmap=PASTIS_CMAP, vmin=0, vmax=20, interpolation='nearest')


def load_agrifm_model(checkpoint_path, device):
    sys.path.insert(0, AGRIFM_REPO)
    sys.path.insert(0, os.path.join(AGRIFM_REPO, 'AgriFM'))
    from mmseg.registry.registry import MODELS
    import AgriFM.models

    full_model = MODELS.build(dict(
        type='MultiUnifiedModel',
        encoders=dict(
            type='MultiModalEncoder',
            encoders_cfg=dict(
                S2=dict(
                    type='PretrainingSwinTransformer3DEncoder',
                    patch_emd_cfg=dict(type='SwinPatchEmbed3D', patch_size=(2,4,4), in_chans=10, embed_dim=128),
                    backbone_cfg=dict(
                        type='SwinTransformer3D', pretrained=None, pretrained2d=False,
                        patch_size=(2,4,4), embed_dim=128, depths=[2,2,18,2],
                        num_heads=[4,8,16,32], window_size=(8,7,7), out_indices=(0,1,2,3),
                        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                        attn_drop_rate=0., drop_path_rate=0.2, patch_norm=False,
                        frozen_stages=-1, use_checkpoint=False,
                        downsample_steps=((2,2,2),(2,2,2),(2,2,2),(2,2,2)),
                        feature_fusion='cat', mean_frame_down=True,
                    ),
                ),
            ),
        ),
        neck=dict(
            type='MultiFusionNeck', embed_dim=2048, in_feature_key=('S2',),
            feature_size=(4,4), out_size=(128,128),
            in_fusion_key_list=({'S2':2048},{'S2':2048},{'S2':2048}),
        ),
        head=dict(
            type='CropFCNHead', num_classes=NUM_CLASSES, embed_dim=2048,
            loss_model=dict(type='CropCEloss'),
        ),
    ))

    ckpt  = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    full_model.load_state_dict(state, strict=False)
    full_model = full_model.to(device).eval()
    logger.info(f"Model loaded from {checkpoint_path}")
    return full_model


@torch.no_grad()
def predict(model, s2_tensor, device):
    s2_norm = (s2_tensor - NORM_MEAN) / (NORM_STD + 1e-6)
    x      = s2_norm.unsqueeze(0).to(device)
    inputs = {'S2': x}
    logits = model(inputs, mode='tensor')
    pred   = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int32)
    conf   = F.softmax(logits, dim=1).max(1).values.squeeze(0).cpu().numpy()
    return pred, conf


NORM_MEAN = torch.tensor([4179.19, 4065.91, 3957.27, 5207.45, 4327.12, 4873.16, 5049.16, 5111.08, 3056.86, 2490.97]).view(1,10,1,1)
NORM_STD  = torch.tensor([4041.52, 3691.00, 3629.33, 2973.52, 3569.73, 3085.92, 2937.56, 2806.04, 1808.30, 1694.20]).view(1,10,1,1)

def load_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        s2    = torch.from_numpy(f['S2'][:].astype(np.float32))
        label = f['label'][:].astype(np.int64)
    return s2, label


def s2_rgb(s2):
    med = np.median(s2.numpy(), axis=0)
    rgb = med[[2,1,0]]
    out = np.zeros((3,rgb.shape[1],rgb.shape[2]), dtype=np.float32)
    for c in range(3):
        lo,hi = np.percentile(rgb[c],2), np.percentile(rgb[c],98)
        out[c] = np.clip((rgb[c]-lo)/(hi-lo+1e-6),0,1)
    return np.transpose(out,(1,2,0))


def compute_metrics(gt, pred):
    iou, f1 = [], []
    for cls in range(NUM_CLASSES):
        tp = np.sum((gt==cls)&(pred==cls))
        fp = np.sum((pred==cls)&(gt!=cls))
        fn = np.sum((pred!=cls)&(gt==cls))
        iou.append(tp/(tp+fp+fn+1e-6))
        p=tp/(tp+fp+1e-6); r=tp/(tp+fn+1e-6)
        f1.append(2*p*r/(p+r+1e-6))
    return np.array(iou), np.array(f1)


def plot_sample(rgb, gt, pred, conf, sid, out_path):
    fig = plt.figure(figsize=(16,4), facecolor=BG)
    gs  = GridSpec(1,4, figure=fig, hspace=0.05, wspace=0.05)
    axes = [fig.add_subplot(gs[0,i]) for i in range(4)]
    fig.suptitle(f'PASTIS · AgriFM Run 2 · Patch {sid}',
                 color='white', fontsize=12, fontweight='bold', y=0.99)
    axes[0].imshow(np.clip(rgb,0,1)); axes[0].set_title('RGB', color='white', fontsize=9)
    axes[1].imshow(gt, **lkw);        axes[1].set_title('Ground Truth', color='white', fontsize=9)
    axes[2].imshow(pred, **lkw);      axes[2].set_title('Prediction', color='white', fontsize=9)
    im = axes[3].imshow(conf, cmap='RdYlGn', vmin=0, vmax=1)
    axes[3].set_title('Confidence', color='white', fontsize=9)
    cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='white')
    for ax in axes: ax.axis('off'); ax.set_facecolor(BG)
    present = sorted(set(np.unique(gt).tolist()+np.unique(pred).tolist()))
    patches = [mpatches.Patch(color=PASTIS_CLASSES[c][1], label=PASTIS_CLASSES[c][0])
               for c in present if c in PASTIS_CLASSES]
    if patches:
        fig.legend(handles=patches, loc='lower center', ncol=min(len(patches),8),
                   fontsize=7, framealpha=0.15, labelcolor='white',
                   facecolor='#1a1a1a', edgecolor='none', bbox_to_anchor=(0.5,-0.12))
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
    plt.close(fig)


def plot_grid(samples, out_path):
    n=len(samples); cols=min(n,4); rows=(n+cols-1)//cols
    fig, axes = plt.subplots(rows*2, cols, figsize=(cols*3.5,rows*7),
                             facecolor=BG, constrained_layout=True)
    fig.suptitle('PASTIS · AgriFM Run 2 · GT vs Pred',
                 color='white', fontsize=14, fontweight='bold')
    def ax(r,c,ro): return axes[r*2+ro][c] if rows>1 else axes[ro][c]
    for i,(gt,pred,sid) in enumerate(samples):
        r,c = divmod(i,cols)
        ax(r,c,0).imshow(gt,**lkw);   ax(r,c,0).set_title(f'Patch {sid}\nGT',color='#aaa',fontsize=8)
        ax(r,c,1).imshow(pred,**lkw); ax(r,c,1).set_title('Pred',color='#aaa',fontsize=8)
        for ro in [0,1]: ax(r,c,ro).axis('off'); ax(r,c,ro).set_facecolor(BG)
    for j in range(n,rows*cols):
        r,c=divmod(j,cols)
        for ro in [0,1]: ax(r,c,ro).axis('off'); ax(r,c,ro).set_facecolor(BG)
    fig.savefig(out_path,dpi=150,bbox_inches='tight',facecolor=BG,edgecolor='none')
    plt.close(fig)
    logger.info(f"Grid → {out_path}")


def plot_confusion(samples, out_path):
    gts   = np.concatenate([g.ravel() for g,p,_ in samples])
    preds = np.concatenate([p.ravel() for g,p,_ in samples])
    cm = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.int64)
    for g,p in zip(gts,preds):
        if 0<=g<NUM_CLASSES and 0<=p<NUM_CLASSES: cm[g,p]+=1
    cm_norm = cm.astype(np.float32)/(cm.sum(1,keepdims=True)+1e-6)
    fig,ax = plt.subplots(figsize=(14,12), facecolor=BG)
    im = ax.imshow(cm_norm, cmap='YlOrRd', aspect='auto')
    labels = [f"{i}\n{PASTIS_CLASSES[i][0][:8]}" for i in range(NUM_CLASSES)]
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(labels,fontsize=6,color='white')
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(labels,fontsize=6,color='white')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Predicted',fontsize=11,color='white')
    ax.set_ylabel('Ground Truth',fontsize=11,color='white')
    ax.set_title('AgriFM Run 2 · Confusion Matrix (normalized by GT)',
                 fontsize=13,color='white',fontweight='bold')
    cbar=plt.colorbar(im,ax=ax); cbar.ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('white')
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    fig.savefig(out_path,dpi=150,bbox_inches='tight',facecolor=BG,edgecolor='none')
    plt.close(fig)
    logger.info(f"Confusion matrix → {out_path}")


def plot_metrics(mean_ious, mean_f1s, out_path):
    fig,axes = plt.subplots(1,2,figsize=(16,6),facecolor=BG)
    fig.suptitle('AgriFM Run 2 · Per-Class Performance on PASTIS',
                 color='white',fontsize=13,fontweight='bold')
    labels=[PASTIS_CLASSES[i][0] for i in range(NUM_CLASSES)]
    x=np.arange(NUM_CLASSES)
    for ax,vals,title,ylabel in [
        (axes[0],mean_ious*100,'Intersection over Union','Mean IoU'),
        (axes[1],mean_f1s*100,'F1 Score','Mean F1 Score'),
    ]:
        colors=['#1D9E75' if v>=60 else '#EF9F27' if v>=30 else '#E24B4A' for v in vals]
        ax.bar(x,vals,color=colors,alpha=0.85,edgecolor='white',linewidth=0.4)
        mean_val=np.nanmean(vals[vals>0]) if np.any(vals>0) else 0
        ax.axhline(mean_val,color='yellow',linestyle='--',linewidth=1.5,
                   label=f'Mean: {mean_val:.1f}%')
        ax.set_xticks(x); ax.set_xticklabels(labels,rotation=45,ha='right',
                                               fontsize=7,color='white')
        ax.set_ylabel(ylabel,color='white'); ax.set_title(title,color='white')
        ax.tick_params(colors='white'); ax.set_facecolor(BG); ax.set_ylim(0,100)
        for sp in ['bottom','left']: ax.spines[sp].set_color('white')
        for sp in ['top','right']:   ax.spines[sp].set_visible(False)
        ax.grid(axis='y',alpha=0.2,color='white')
        ax.legend(labelcolor='white',fontsize=8)
    fig.patch.set_facecolor(BG)
    plt.tight_layout()
    fig.savefig(out_path,dpi=150,bbox_inches='tight',facecolor=BG,edgecolor='none')
    plt.close(fig)
    logger.info(f"Metrics → {out_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out    = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    model  = load_agrifm_model(str(CHECKPOINT_PATH), device)
    test_ids = np.loadtxt(TEST_LIST,dtype=str).tolist()[:args.num_samples]
    all_ious,all_f1s,grid_data = [],[],[]
    for idx,pid in enumerate(test_ids):
        h5 = DATA_DIR/f'{pid}.h5'
        if not h5.exists(): logger.warning(f"Missing: {h5}"); continue
        logger.info(f"[{idx+1}/{len(test_ids)}] {pid}")
        s2,gt   = load_h5(h5)
        pred,conf = predict(model,s2,device)
        rgb     = s2_rgb(s2)
        iou,f1  = compute_metrics(gt,pred)
        all_ious.append(iou); all_f1s.append(f1)
        grid_data.append((gt,pred,pid))
        plot_sample(rgb,gt,pred,conf,pid,out/f'sample_{idx:03d}_{pid}.png')
    if not all_ious: logger.error("No samples processed"); return
    mean_ious = np.array(all_ious).mean(0)
    mean_f1s  = np.array(all_f1s).mean(0)
    if args.plot_grid: plot_grid(grid_data, out/'predictions_grid.png')
    plot_confusion(grid_data, out/'confusion_matrix.png')
    plot_metrics(mean_ious, mean_f1s, out/'class_performance.png')
    print(f"\nmIoU: {mean_ious.mean()*100:.2f}% | mF1: {mean_f1s.mean()*100:.2f}%")
    print(f"Output: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--output-dir', type=str,
                        default=str(BASE_OUTPUT_DIR/'visualizations'/'agrifm_run2_pastis'))
    parser.add_argument('--no-grid', dest='plot_grid', action='store_false', default=True)
    main(parser.parse_args())
