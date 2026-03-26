# Geobreeze Crop Mapping Extensions

This repository contains extensions to [geobreeze](https://github.com/geobreeze/geobreeze) for crop type mapping evaluation on the PASTIS-R and GEO-Bench m-SA-crop-type datasets.

## Models Integrated into Geobreeze

| Model | File | Config |
|---|---|---|
| AgriFM | `agrifm.py` | `configs/agrifm.yaml` |
| SatlasPretrain | `satlaspretrain_initial_mean.py` / `satlaspretrain_new_temporal_stack.py` | `configs/satlaspretrain.yaml` |
| Prithvi-EO-2.0-600M-TL | `prithvi.py` | `configs/prithvi_600m_tl.yaml` |
| SSL4EO-S12 (DINO) | `ssl4eo.py` | `configs/ssl4eo_s2_dino.yaml` |
| OlmoEarth | `olmoearth.py` | `configs/olmoearth_base.yaml` |

## Results on PASTIS-R Test Set (mIoU)

All geobreeze experiments use frozen backbone + UPerNet segmentation head (segm_frozen_backbone mode), 50 epochs, AdamW. All models below use **temporal_strategy=mean** (single image, mean-collapsed across timestamps).

| Model | Modality | mIoU |
|---|---|---|
| OlmoEarth | S2 | 25.77% |
| SatlasPretrain | S2 | 23.66% |
| Prithvi-EO-2.0-600M-TL | S2 | 21.59% |
| SSL4EO-S12 DINO | S2 | 21.25% |
| Panopticon | S1+S2 | 21.07% |
| CROMA | S2 | 21.07% |
| AgriFM (geobreeze) | S2 | ~9% |

> **CropSTS published SoTA (parcel-aware): 39.09% mIoU**

### Effect of Temporal Strategy

The table below compares models tested with both mean-collapsed and temporal stack strategies. Stack passes the full time series (T=46 timestamps) to the model instead of collapsing to a single image.

| Model | Temporal Strategy | mIoU | Δ vs Mean |
|---|---|---|---|
| SatlasPretrain | mean | 23.66% | — |
| SatlasPretrain | stack + maxpool | 25.29% | +1.63pp |
| AgriFM (geobreeze) | mean | ~9% | — |
| AgriFM (geobreeze) | stack (T=32 sampled) | 11.09% | +2pp |

> Note: SatlasPretrain stack encodes each frame independently then max-pools over T, similar to its own multi-image pretraining strategy. AgriFM stack uniformly samples 32 frames from 46 timestamps.

### AgriFM Standalone Temporal Training (Outside Geobreeze)

Training AgriFM end-to-end with its native temporal architecture (outside geobreeze) gives substantially better results, demonstrating the value of full temporal modeling:

| Model | Temporal Strategy | mIoU |
|---|---|---|
| AgriFM standalone | Full T=32 | **45.08%** |
| CropSTS (published SoTA, parcel-aware) | Full temporal | 39.09% |

> AgriFM standalone achieves **+6pp over published SoTA** without using parcel boundary information. See `agrifm-pastis-outside-geobreeze/` for training scripts.

## Results on GEO-Bench m-SA-crop-type

| Model | Results |
|---|---|
| Panopticon | `geobench-m-SA-crop-type/panopticon_m_SA_results.csv` |
| CROMA | `geobench-m-SA-crop-type/croma_m_SA_results.csv` |

## Repository Structure
```
├── agrifm.py                                   # AgriFM geobreeze wrapper
├── satlaspretrain_initial_mean.py              # SatlasPretrain (mean temporal)
├── satlaspretrain_new_temporal_stack.py        # SatlasPretrain (stack temporal)
├── prithvi.py                                  # Prithvi-EO-2.0-600M-TL wrapper
├── ssl4eo.py                                   # SSL4EO-S12 DINO wrapper
├── olmoearth.py                                # OlmoEarth wrapper
├── pastis_initial_temporal_strategy_mean.py    # PASTIS dataset (mean)
├── pastis_new_temporal_strategy_stack.py       # PASTIS dataset (stack)
├── geobench.py                                 # GEO-Bench dataset
├── mmcv_mock_module.py                         # mmcv mock (required for AgriFM)
├── configs/                                    # Model and dataset configs
├── results/                                    # Test set results CSVs
├── visualization/                              # Visualization scripts
├── geobench-m-SA-crop-type/                    # GEO-Bench results and scripts
└── agrifm-pastis-outside-geobreeze/            # AgriFM standalone temporal training
```

## Usage

### Install geobreeze
```bash
git clone https://github.com/geobreeze/geobreeze
cd geobreeze
pip install -e .
```

### Copy model files
```bash
cp agrifm.py satlaspretrain_new_temporal_stack.py prithvi.py ssl4eo.py olmoearth.py geobreeze/geobreeze/models/
cp configs/*.yaml geobreeze/geobreeze/config/model/base/
cp pastis_new_temporal_strategy_stack.py geobreeze/geobreeze/datasets/pastis.py
cp geobench.py geobreeze/geobreeze/datasets/
```

### Run experiments
```bash
# PASTIS with Prithvi-EO-2.0-600M-TL
python geobreeze/main.py +model=base/prithvi_600m_tl +data=pastis-s2 +optim=segmentation output_dir=outputs/prithvi_pastis dl.batch_size=4

# PASTIS with SSL4EO-S12 DINO
python geobreeze/main.py +model=base/ssl4eo_s2_dino +data=pastis-s2 +optim=segmentation output_dir=outputs/ssl4eo_pastis

# PASTIS with OlmoEarth
python geobreeze/main.py +model=base/olmoearth_base +data=pastis-s2 +optim=segmentation output_dir=outputs/olmoearth_pastis

# PASTIS with SatlasPretrain (temporal stack)
python geobreeze/main.py +model=base/satlaspretrain +data=pastis-s2-stack +optim=segmentation output_dir=outputs/satlaspretrain_pastis_stack dl.batch_size=4

# GEO-Bench m-SA-crop-type with Panopticon
python geobreeze/main.py +model=base/panopticon +data=m-SA-crop-type +optim=segmentation output_dir=outputs/panopticon_SA
```

## Requirements

- `mmcv_mock_module.py` — mock for mmcv (required for AgriFM imports, zero effect on results)
- `DATASETS_DIR` environment variable pointing to dataset directory
- PASTIS-R data at `$DATASETS_DIR/pastis-r`
- GEO-Bench data at `$GEO_BENCH_DIR`
