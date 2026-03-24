"""PASTIS-R Temporal Segmentation Dataset."""

import json
import logging
import os
import numpy as np
import torch
import kornia.augmentation as K
from pathlib import Path
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def pastis_temporal_collate_fn(batch):
    """Custom collate for stack mode — pads time dimension to max T in batch."""
    imgs_list, label_list = zip(*batch)
    imgs_list = [x['imgs'] for x in imgs_list]
    T_max = max(x.shape[0] for x in imgs_list)
    padded = []
    for x in imgs_list:
        T, C, H, W = x.shape
        if T < T_max:
            pad = torch.zeros(T_max - T, C, H, W, dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        padded.append(x)
    imgs_batch = torch.stack(padded, dim=0)  # (B, T_max, C, H, W)
    labels_batch = torch.stack(label_list, dim=0)
    # rebuild x_dict using first sample for metadata
    x_dict = {k: v for k, v in batch[0][0].items() if k != 'imgs'}
    x_dict['imgs'] = imgs_batch
    return x_dict, labels_batch

# PASTIS-R fold assignments:
# Folds 1-4 -> train, Fold 5 -> val, Fold 5 (test subset) -> test
# Standard PASTIS split: train=folds 1-4, val=fold 5, test=fold 5
SPLIT_FOLDS = {
    'train': [1, 2, 3, 4],
    'val':   [5],
    'test':  [5],
}


class PASTIS(Dataset):
    """PASTIS-R dataset for temporal crop type segmentation.

    Data layout (flat, no split subdirectories):
        root/DATA_S2/S2_<id>.npy      shape (T, 10, 128, 128)  int16
        root/DATA_S1A/S1A_<id>.npy    shape (T,  3, 128, 128)  float32
        root/DATA_S1D/S1D_<id>.npy    shape (T,  3, 128, 128)  float32
        root/ANNOTATIONS/TARGET_<id>.npy  shape (3, 128, 128)  uint8
        root/NORM_S2_patch.json
        root/NORM_S1A_patch.json
        root/NORM_S1D_patch.json
        root/metadata.geojson

    Args:
        root: Path to the PASTIS-R root directory.
        split: One of 'train', 'val', 'test'.
        modality: One of 's2', 'all'.
        temporal_strategy: How to collapse the time axis — 'mean', 'max', 'first', 'random'.
        normalize: Whether to z-score normalise using the per-fold statistics.
        transform: List of kornia augmentations (applied to image + mask jointly).
    """

    valid_modalities          = ['s2', 'all']
    valid_temporal_strategies = ['mean', 'max', 'first', 'random', 'stack']

    def __init__(
        self,
        root: str,
        split: str = 'train',
        modality: str = 's2',
        temporal_strategy: str = 'mean',
        normalize: bool = True,
        transform: list = None,
        seed: int = 42,
        **kwargs,          # absorb extra hydra keys gracefully
    ):
        assert split    in SPLIT_FOLDS,                  f"split must be one of {list(SPLIT_FOLDS)}"
        assert modality in self.valid_modalities,        f"modality must be one of {self.valid_modalities}"
        assert temporal_strategy in self.valid_temporal_strategies, \
               f"temporal_strategy must be one of {self.valid_temporal_strategies}"

        self.root              = Path(root).expanduser()
        self.split             = split
        self.modality          = modality
        self.temporal_strategy = temporal_strategy
        self.normalize         = normalize
        self.seed              = seed

        # Kornia augmentation pipeline (image + mask)
        self.trf = K.AugmentationSequential(
            *(transform or []),
            data_keys=["input", "mask"],
            same_on_batch=True,
        ) if transform else None

        self._load_metadata()
        self._load_norm_stats()

        # Attributes expected by geobreeze main.py
        # S2 bands: B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 (gaussian.mu in nm)
        S2_CHN_IDS = torch.tensor([
            492.997, 559.599, 664.630, 704.006, 740.552,
            782.419, 827.539, 864.780, 1613.862, 2203.618
        ])
        # S1: VV=-1, VH=-2, HH=-3 (x2 for S1A and S1D)
        S1_CHN_IDS = torch.tensor([-1.0, -2.0, -3.0, -1.0, -2.0, -3.0])
        S2_GSD = torch.tensor([10., 10., 10., 20., 20., 20., 10., 20., 20., 20.])
        S1_GSD = torch.tensor([10., 10., 10., 10., 10., 10.])

        if modality == 's2':
            self.band_ids = list(range(10))
            self.chn_ids  = S2_CHN_IDS
            self.gsd      = S2_GSD
        else:  # 'all'
            self.band_ids = list(range(16))  # 10 S2 + 3 S1A + 3 S1D
            self.chn_ids  = torch.cat([S2_CHN_IDS, S1_CHN_IDS])
            self.gsd      = torch.cat([S2_GSD, S1_GSD])
        self.num_classes = 21  # PASTIS-R: 20 crop types + background

        logger.info(
            f"PASTIS {split}: {len(self.patch_ids)} patches | "
            f"modality={modality} | temporal_strategy={temporal_strategy} | "
            f"num_channels={len(self.band_ids)} | num_classes={self.num_classes}"
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_metadata(self):
        """Parse metadata.geojson and select patches for this split."""
        meta_path = self.root / 'metadata.geojson'
        with open(meta_path) as f:
            geo = json.load(f)

        target_folds = set(SPLIT_FOLDS[self.split])

        # val and test both live in fold 5; split them 50/50 by patch index
        all_fold5, others = [], []
        for feat in geo['features']:
            props = feat['properties']
            fold  = props['Fold']
            pid   = int(props['ID_PATCH'])
            if fold in target_folds:
                if fold == 5:
                    all_fold5.append(pid)
                else:
                    others.append(pid)

        all_fold5.sort()

        if self.split == 'train':
            self.patch_ids = sorted(others)
        elif self.split == 'val':
            self.patch_ids = all_fold5[: len(all_fold5) // 2]
        else:  # test
            self.patch_ids = all_fold5[len(all_fold5) // 2 :]

    def _load_norm_stats(self):
        """Load per-fold normalisation statistics from NORM_*.json files."""
        folds = SPLIT_FOLDS[self.split]
        # Use fold 1 stats for validation/test (standard practice)
        fold_key = f'Fold_{folds[0]}'

        def _load(fname):
            with open(self.root / fname) as f:
                d = json.load(f)
            entry = d[fold_key]
            mean  = torch.tensor(entry['mean'], dtype=torch.float32)
            std   = torch.tensor(entry['std'],  dtype=torch.float32)
            return mean, std

        self.s2_mean, self.s2_std = _load('NORM_S2_patch.json')

        if self.modality == 'all':
            s1a_mean, s1a_std = _load('NORM_S1A_patch.json')
            s1d_mean, s1d_std = _load('NORM_S1D_patch.json')
            # S1A and S1D each have 3 channels; concatenate → 6 channels
            self.s1_mean = torch.cat([s1a_mean, s1d_mean])
            self.s1_std  = torch.cat([s1a_std,  s1d_std])

    # ------------------------------------------------------------------
    # Temporal aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, arr: np.ndarray) -> np.ndarray:
        """Collapse (T, C, H, W) → (C, H, W) using self.temporal_strategy."""
        if self.temporal_strategy == 'mean':
            return arr.mean(axis=0)
        elif self.temporal_strategy == 'max':
            return arr.max(axis=0)
        elif self.temporal_strategy == 'first':
            return arr[0]
        elif self.temporal_strategy == 'random':
            rng = np.random.RandomState(self.seed)
            return arr[rng.randint(0, arr.shape[0])]
        else:  # 'stack'
            return arr  # keep (T, C, H, W) as-is

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        pid = self.patch_ids[idx]

        # --- Load S2 -------------------------------------------------------
        s2 = np.load(self.root / 'DATA_S2'  / f'S2_{pid}.npy').astype(np.float32)
        # shape: (T, 10, 128, 128)
        s2 = self._aggregate(s2)           # (10, 128, 128)
        s2 = torch.from_numpy(s2)

        if self.modality == 'all':
            s1a = np.load(self.root / 'DATA_S1A' / f'S1A_{pid}.npy').astype(np.float32)
            s1d = np.load(self.root / 'DATA_S1D' / f'S1D_{pid}.npy').astype(np.float32)
            s1a = self._aggregate(s1a)     # (3, 128, 128)
            s1d = self._aggregate(s1d)     # (3, 128, 128)
            s1  = torch.from_numpy(np.concatenate([s1a, s1d], axis=0))  # (6, 128, 128)

        # --- Normalise ------------------------------------------------------
        if self.normalize:
            if self.temporal_strategy == 'stack':
                s2 = (s2 - self.s2_mean[None, :, None, None]) / (self.s2_std[None, :, None, None] + 1e-6)
                if self.modality == 'all':
                    s1 = (s1 - self.s1_mean[None, :, None, None]) / (self.s1_std[None, :, None, None] + 1e-6)
            else:
                s2 = (s2 - self.s2_mean[:, None, None]) / (self.s2_std[:, None, None] + 1e-6)
                if self.modality == 'all':
                    s1 = (s1 - self.s1_mean[:, None, None]) / (self.s1_std[:, None, None] + 1e-6)

        # --- Concatenate modalities ----------------------------------------
        if self.modality == 's2':
            imgs = s2                                   # (10, H, W)
        else:
            imgs = torch.cat([s2, s1], dim=0)          # (16, H, W)

        # --- Load label (channel 0 = semantic class) -----------------------
        target = np.load(self.root / 'ANNOTATIONS' / f'TARGET_{pid}.npy')
        label  = torch.from_numpy(target[0].astype(np.int64))  # (128, 128)

        # --- Augmentations -------------------------------------------------
        if self.trf is not None and self.temporal_strategy != 'stack':
            imgs, label = self.trf(
                imgs.unsqueeze(0),           # (1, C, H, W)
                label.unsqueeze(0).unsqueeze(0).float(),  # (1, 1, H, W)
            )
            imgs  = imgs.squeeze(0)
            label = label.squeeze(0).squeeze(0).long()

        x = dict(imgs=imgs, band_ids=torch.tensor(self.band_ids), chn_ids=self.chn_ids, gsd=self.gsd)
        return x, label
