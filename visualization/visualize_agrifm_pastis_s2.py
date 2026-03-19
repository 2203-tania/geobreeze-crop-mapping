"""
Visualization script for AgriF Foundation Model on PASTIS-S2.

Generates detailed visualizations and analysis of predictions from the AgriF model
trained on PASTIS Sentinel-2 crop classification, including:
  - Per-sample prediction maps (RGB, GT, Pred)
  - Comparison grids
  - Per-class performance metrics (IoU, F1)
  - Confusion matrices

Usage:
    # Quick visualization with defaults (16 test samples)
    python visualize_agrifm_pastis_s2.py

    # Custom runs
    python visualize_agrifm_pastis_s2.py --num-samples 32 --split test
    python visualize_agrifm_pastis_s2.py --output-dir ./custom_viz --split val
    python visualize_agrifm_pastis_s2.py --no-grid
    
Note: Requires DATASETS_DIR environment variable to be set to the PASTIS dataset root.
"""

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
import seaborn as sns
from pathlib import Path
from datetime import datetime

# PyTorch 2.6 fix: force weights_only=False for trusted local checkpoints
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# =========================================================================
# CONFIGURATION
# =========================================================================

# Fixed paths for agrifm_pastis_s2
BASE_OUTPUT_DIR = Path('/mnt/tania/geobreeze/outputs/agrifm_pastis_s2')
CHECKPOINT_PATH = BASE_OUTPUT_DIR / 'lr=3.125e-06_bsz=8_e=50/checkpoints/last.ckpt'

# -------------------------------------------------------------------------
# PASTIS-R class definitions (21 classes including void label)
# From PASTIS dataset: 20 crop types + background + void
# -------------------------------------------------------------------------
PASTIS_CLASSES = {
    0:  ('Background',                          '#000000'),
    1:  ('Meadow',                              '#7CFC00'),
    2:  ('Soft Winter Wheat',                   '#FFD700'),
    3:  ('Corn',                                '#FF8C00'),
    4:  ('Winter Barley',                       '#DAA520'),
    5:  ('Winter Rapeseed',                     '#ADFF2F'),
    6:  ('Spring Barley',                       '#F0E68C'),
    7:  ('Sunflower',                           '#FFA500'),
    8:  ('Grapevine',                           '#8B0000'),
    9:  ('Beet',                                '#FF69B4'),
    10: ('Winter Triticale',                    '#BDB76B'),
    11: ('Winter Durum Wheat',                  '#EEE8AA'),
    12: ('Fruits, Vegetables, Flowers',        '#228B22'),
    13: ('Potatoes',                            '#D2691E'),
    14: ('Leguminous Fodder',                   '#90EE90'),
    15: ('Soybeans',                            '#6B8E23'),
    16: ('Orchard',                             '#006400'),
    17: ('Mixed Cereals',                       '#F5DEB3'),
    18: ('Sorghum',                             '#CD853F'),
    19: ('Void Label',                          '#808080'),
    20: ('Unknown',                             '#C0C0C0'),
}

CMAP_COLORS = [PASTIS_CLASSES[i][1] for i in range(21)]
PASTIS_CMAP = ListedColormap(CMAP_COLORS)


# =========================================================================
# UTILITIES
# =========================================================================

def load_lightning_task(checkpoint_path: str, device: str):
    """Load the full LightningSegmentationTask from checkpoint."""
    sys.path.insert(0, str(Path(__file__).parent))

    geobreeze_root = Path(__file__).parent / 'geobreeze'
    if geobreeze_root.exists():
        sys.path.insert(0, str(geobreeze_root.parent))

    from geobreeze.engine.lightning_task import LightningSegmentationTask
    from geobreeze.models.agrifm import AgriFM
    from omegaconf import OmegaConf, DictConfig

    # Load checkpoint to extract saved hyper_parameters
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['hyper_parameters']['cfg']
    num_classes = ckpt['hyper_parameters']['num_classes']
    num_channels = ckpt['hyper_parameters']['num_channels']

    # Convert cfg dict → OmegaConf if needed
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    # Build AgriFM encoder using saved config
    model_cfg = cfg.model
    logger.info(f"Building AgriFM encoder with config")
    encoder = AgriFM(
        blk_indices=list(model_cfg.blk_indices) if hasattr(model_cfg, 'blk_indices') else [0, 0, 0, 0],
        checkpoint_path=model_cfg.checkpoint_path,
        agrifm_repo_path=model_cfg.agrifm_repo_path,
        image_resolution=model_cfg.image_resolution,
        embed_dim=model_cfg.embed_dim,
        patch_size=model_cfg.patch_size,
    )

    # Load checkpoint with encoder passed in
    task = LightningSegmentationTask.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        strict=False,
        cfg=cfg,
        encoder=encoder,
        num_classes=num_classes,
        num_channels=num_channels,
    )
    task = task.to(device)
    task.eval()
    logger.info("Model loaded and set to eval mode.")
    return task


@torch.no_grad()
def predict(task, batch_dict, device):
    """Run forward pass and return prediction and confidence maps.
    
    Args:
        batch_dict: dict from dataset with 'imgs', 'chn_ids', 'gsd', 'band_ids'
    """
    imgs = batch_dict['imgs'].unsqueeze(0).to(device)
    chn_ids = batch_dict['chn_ids'].unsqueeze(0).to(device) if 'chn_ids' in batch_dict else None
    gsd = batch_dict['gsd'].unsqueeze(0).to(device) if 'gsd' in batch_dict else None
    band_ids = batch_dict['band_ids'].to(device) if 'band_ids' in batch_dict else None
    
    x = {'imgs': imgs}
    if chn_ids is not None:
        x['chn_ids'] = chn_ids
    if gsd is not None:
        x['gsd'] = gsd
    if band_ids is not None:
        x['band_ids'] = band_ids
    
    out = task(x)
    # Handle both single output and tuple output
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    confidence = probs.max(axis=0)
    
    return pred, probs, confidence


def s2_to_rgb(imgs_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized S2 tensor (C, H, W) to displayable uint8 RGB.
    
    PASTIS S2 band order: B02(blue), B03(green), B04(red), ...
    Indices: 0=B02(blue), 1=B03(green), 2=B04(red)
    For RGB we want: [2, 1, 0] = [red, green, blue]
    """
    arr = imgs_tensor.numpy()  # (C, H, W)
    # Use B04 (Red), B03 (Green), B02 (Blue) for RGB
    rgb = arr[[2, 1, 0], :, :]  # B04, B03, B02 → R, G, B
    
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        lo, hi = np.percentile(rgb[c], 2), np.percentile(rgb[c], 98)
        out[c] = np.clip((rgb[c] - lo) / (hi - lo + 1e-6), 0, 1)
    
    return np.transpose(out, (1, 2, 0))  # (H, W, 3)


def compute_per_class_metrics(gt, pred, num_classes=21):
    """Compute IoU and F1 per class."""
    iou_per_class = []
    f1_per_class = []
    
    for cls in range(num_classes):
        gt_mask = (gt == cls)
        pred_mask = (pred == cls)
        
        tp = np.sum(gt_mask & pred_mask)
        fp = np.sum(pred_mask & ~gt_mask)
        fn = np.sum(~pred_mask & gt_mask)
        
        # IoU
        iou = tp / (tp + fp + fn + 1e-6)
        iou_per_class.append(iou)
        
        # F1
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        f1_per_class.append(f1)
    
    return np.array(iou_per_class), np.array(f1_per_class)


# =========================================================================
# VISUALIZATION FUNCTIONS
# =========================================================================

def plot_sample_detailed(rgb, gt, pred, confidence, patch_id, out_path):
    """Save a 4-panel figure: RGB | GT | Pred | Confidence."""
    fig = plt.figure(figsize=(16, 4), facecolor='#0d0d0d')
    gs = GridSpec(1, 4, figure=fig, hspace=0.05, wspace=0.05)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    
    fig.suptitle(f'PASTIS Patch {patch_id} · AgriF Prediction',
                 color='white', fontsize=13, fontweight='bold', y=0.98)

    lkw = dict(cmap=PASTIS_CMAP, vmin=0, vmax=20, interpolation='nearest')

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title('Sentinel-2 RGB', color='white', fontsize=10)

    # Ground Truth
    axes[1].imshow(gt, **lkw)
    axes[1].set_title('Ground Truth', color='white', fontsize=10)

    # Prediction
    axes[2].imshow(pred, **lkw)
    axes[2].set_title('Prediction', color='white', fontsize=10)

    # Confidence Map
    im = axes[3].imshow(confidence, cmap='RdYlGn', vmin=0, vmax=1)
    axes[3].set_title('Confidence', color='white', fontsize=10)
    cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label('Max Prob', color='white', fontsize=8)

    for ax in axes:
        ax.axis('off')
        ax.set_facecolor('#0d0d0d')

    # Legend
    present = np.unique(np.concatenate([gt.ravel(), pred.ravel()]))
    present = [int(c) for c in present if c in PASTIS_CLASSES]
    patches = [
        mpatches.Patch(color=PASTIS_CLASSES[c][1], label=PASTIS_CLASSES[c][0])
        for c in present
    ]
    if patches:
        fig.legend(handles=patches, loc='lower center', ncol=min(len(patches), 7),
                   fontsize=7, framealpha=0.15, labelcolor='white',
                   facecolor='#1a1a1a', edgecolor='none',
                   bbox_to_anchor=(0.5, -0.08))

    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0d0d0d', edgecolor='none')
    plt.close(fig)


def plot_comparison_grid(samples, out_path):
    """
    Plot GT vs Pred comparison grid for multiple samples.
    samples: list of (gt, pred, patch_id)
    """
    n = len(samples)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows * 2, cols,
                             figsize=(cols * 3.5, rows * 7),
                             facecolor='#0d0d0d', constrained_layout=True)
    fig.suptitle('PASTIS · AgriF Predictions (GT vs Pred)',
                 color='white', fontsize=16, fontweight='bold')

    lkw = dict(cmap=PASTIS_CMAP, vmin=0, vmax=20, interpolation='nearest')

    for i, (gt, pred, pid) in enumerate(samples):
        row, col = divmod(i, cols)
        ax_gt = axes[row * 2][col] if rows > 1 else axes[0][col]
        ax_pred = axes[row * 2 + 1][col] if rows > 1 else axes[1][col]

        ax_gt.imshow(gt, **lkw)
        ax_gt.set_title(f'Patch {pid}\nGT', color='#aaaaaa', fontsize=8)
        ax_gt.axis('off')
        ax_gt.set_facecolor('#0d0d0d')

        ax_pred.imshow(pred, **lkw)
        ax_pred.set_title('Pred', color='#aaaaaa', fontsize=8)
        ax_pred.axis('off')
        ax_pred.set_facecolor('#0d0d0d')

    # Hide unused axes
    total_slots = rows * cols
    for j in range(n, total_slots):
        row, col = divmod(j, cols)
        for r_off in [0, 1]:
            ax = axes[row * 2 + r_off][col] if rows > 1 else axes[r_off][col]
            ax.axis('off')
            ax.set_facecolor('#0d0d0d')

    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0d0d0d', edgecolor='none')
    plt.close(fig)
    logger.info(f"Grid saved → {out_path}")


def plot_class_performance(all_ious, all_f1s, out_path):
    """Plot per-class IoU and F1 scores."""
    mean_iou = np.mean(all_ious, axis=0)
    mean_f1 = np.mean(all_f1s, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d0d0d')
    fig.suptitle('Per-Class Performance (AgriF on PASTIS)',
                 color='white', fontsize=14, fontweight='bold')

    class_names = [PASTIS_CLASSES[i][0] for i in range(21)]
    x = np.arange(len(class_names))
    width = 0.6

    # IoU
    ax = axes[0]
    bars = ax.bar(x, mean_iou, width, color='#2E86AB', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Class', color='white', fontsize=11)
    ax.set_ylabel('Mean IoU', color='white', fontsize=11)
    ax.set_title('Intersection over Union', color='white', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8, color='white')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.2, color='white')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')

    # F1
    ax = axes[1]
    bars = ax.bar(x, mean_f1, width, color='#A23B72', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Class', color='white', fontsize=11)
    ax.set_ylabel('Mean F1 Score', color='white', fontsize=11)
    ax.set_title('F1 Score', color='white', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8, color='white')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.2, color='white')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0d0d0d', edgecolor='none')
    plt.close(fig)
    logger.info(f"Class performance saved → {out_path}")


def plot_confusion_matrix(all_gt, all_pred, out_path):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(all_gt.ravel(), all_pred.ravel(), labels=np.arange(21))
    
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#0d0d0d')
    
    sns.heatmap(cm, annot=False, fmt='d', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Count'}, square=True)
    
    ax.set_xlabel('Predicted', color='white', fontsize=12)
    ax.set_ylabel('Ground Truth', color='white', fontsize=12)
    ax.set_title('Confusion Matrix (AgriF on PASTIS)', 
                 color='white', fontsize=13)
    
    class_names = [PASTIS_CLASSES[i][0] for i in range(21)]
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, rotation=0, fontsize=8)
    
    ax.set_facecolor('#0d0d0d')
    fig.patch.set_facecolor('#0d0d0d')
    
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0d0d0d', edgecolor='none')
    plt.close(fig)
    logger.info(f"Confusion matrix saved → {out_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize AgriF PASTIS-S2 model predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--output-dir', type=str, 
                        default=str(BASE_OUTPUT_DIR / 'visualizations'),
                        help='Where to save visualizations')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to visualize')
    parser.add_argument('--num-samples', type=int, default=16,
                        help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--no-grid', action='store_true',
                        help='Skip summary grid visualization')
    parser.add_argument('--no-metrics', action='store_true',
                        help='Skip per-class metrics and confusion matrix')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir = out_dir / 'per_sample'
    per_sample_dir.mkdir(exist_ok=True)

    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")

    # Check checkpoint exists
    if not CHECKPOINT_PATH.exists():
        logger.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)

    # Load model
    task = load_lightning_task(str(CHECKPOINT_PATH), args.device)

    # Load dataset
    try:
        from geobreeze.datasets.pastis import PASTIS
        
        # Use config path or environment variable
        pastis_root = os.environ.get('PASTIS_ROOT', '/mnt/tania/pastis-r')
        
        dataset = PASTIS(
            root=pastis_root,
            split=args.split,
            modality='s2',
            temporal_strategy='mean',
            normalize=True,
        )
        logger.info(f"Dataset loaded: {len(dataset)} samples in '{args.split}' split")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error(f"Make sure DATASETS_DIR environment variable is set to PASTIS root")
        sys.exit(1)

    # Sample indices
    n = min(args.num_samples, len(dataset))
    indices = np.random.choice(len(dataset), size=n, replace=False)

    # Run inference
    collected_samples = []
    all_ious = []
    all_f1s = []
    all_gt = []
    all_pred = []

    logger.info(f"Running inference on {n} samples...")
    for rank, idx in enumerate(indices):
        x_dict, label = dataset[idx]
        patch_id = dataset.patch_ids[idx]

        pred, probs, confidence = predict(task, x_dict, args.device)
        gt = label.numpy()
        rgb = s2_to_rgb(x_dict['imgs'])

        # Per-sample figure
        sample_path = per_sample_dir / f'patch_{patch_id}.png'
        plot_sample_detailed(rgb, gt, pred, confidence, patch_id, sample_path)

        collected_samples.append((gt, pred, patch_id))
        all_gt.append(gt)
        all_pred.append(pred)

        # Metrics
        iou, f1 = compute_per_class_metrics(gt, pred)
        all_ious.append(iou)
        all_f1s.append(f1)

        logger.info(f"[{rank+1}/{n}] Patch {patch_id} (mIoU: {iou.mean():.4f}) → {sample_path}")

    # Summary grid
    if not args.no_grid:
        grid_path = out_dir / f'predictions_grid_{args.split}.png'
        plot_comparison_grid(collected_samples, grid_path)

    # Per-class metrics
    if not args.no_metrics:
        metrics_path = out_dir / f'class_performance_{args.split}.png'
        plot_class_performance(np.array(all_ious), np.array(all_f1s), metrics_path)

        # Confusion matrix
        all_gt_concat = np.concatenate(all_gt)
        all_pred_concat = np.concatenate(all_pred)
        cm_path = out_dir / f'confusion_matrix_{args.split}.png'
        plot_confusion_matrix(all_gt_concat, all_pred_concat, cm_path)

    # Summary stats
    logger.info("\n" + "="*70)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*70)
    all_ious = np.array(all_ious)
    all_f1s = np.array(all_f1s)
    logger.info(f"Mean mIoU: {all_ious.mean():.4f} ± {all_ious.mean(axis=0).std():.4f}")
    logger.info(f"Mean mF1:  {all_f1s.mean():.4f} ± {all_f1s.mean(axis=0).std():.4f}")
    logger.info(f"Split: {args.split} | Device: {args.device}")
    logger.info("="*70)
    logger.info(f"Full results saved to: {out_dir}")


if __name__ == '__main__':
    main()
