import os
import numpy as np
import h5py
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm

PASTIS_ROOT = "/mnt/tania/pastis-r"
OUTPUT_ROOT = "/mnt/tania/pastis_agrifm"
NUM_FRAMES  = 32

def sample_frames(T, num_frames):
    if T >= num_frames:
        return np.linspace(0, T - 1, num_frames, dtype=int)
    else:
        return np.concatenate([np.arange(T), np.full(num_frames - T, T - 1)]).astype(int)

def convert():
    pastis = Path(PASTIS_ROOT)
    out_h5   = Path(OUTPUT_ROOT) / 'h5_samples'
    out_list = Path(OUTPUT_ROOT) / 'data_list'
    out_h5.mkdir(parents=True, exist_ok=True)
    out_list.mkdir(parents=True, exist_ok=True)

    meta = gpd.read_file(pastis / 'metadata.geojson')
    fold5 = sorted(meta[meta['Fold'] == 5]['ID_PATCH'].tolist())
    mid = len(fold5) // 2

    splits = {
        'train': meta[meta['Fold'].isin([1,2,3,4])]['ID_PATCH'].tolist(),
        'val':   fold5[:mid],
        'test':  fold5[mid:],
    }

    for split, ids in splits.items():
        print(f"\nConverting {split}: {len(ids)} patches...")
        valid_ids = []
        for patch_id in tqdm(ids):
            s2_path  = pastis / 'DATA_S2' / f'S2_{patch_id}.npy'
            lbl_path = pastis / 'ANNOTATIONS' / f'TARGET_{patch_id}.npy'
            if not s2_path.exists() or not lbl_path.exists():
                continue
            s2 = np.load(s2_path).astype(np.float32)
            idx = sample_frames(s2.shape[0], NUM_FRAMES)
            s2 = s2[idx]
            label = np.load(lbl_path)[0].astype(np.int64)
            h5_path = out_h5 / f'{patch_id}.h5'
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('S2',    data=s2,    compression='gzip', compression_opts=4)
                f.create_dataset('label', data=label, compression='gzip', compression_opts=4)
            valid_ids.append(str(patch_id))

        with open(out_list / f'{split}.txt', 'w') as f:
            f.write('\n'.join(valid_ids))
        print(f"  Written {len(valid_ids)} samples")

    print(f"\nDone. Output at: {OUTPUT_ROOT}")

if __name__ == '__main__':
    convert()
