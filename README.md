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

All geobreeze experiments use frozen backbone + UPerNet segmentation head (segm_frozen_backbone mode), 50 epochs, AdamW, batch_size=4-8.

| Model | Modality | Temporal Strategy | mIoU |
|---|---|---|---|
| AgriFM (standalone, outside geobreeze) | S2 | Full T=32 | 45.08% |
| OlmoEarth | S2 | mean | 25.77% |
| SatlasPretrain | S2 | stack + maxpool | 25.29% |
| Prithvi-EO-2.0-600M-TL | S2 | mean (T=1) | 21.59% |
| SSL4EO-S12 DINO | S2 | mean | 21.25% |
| SatlasPretrain | S2 | mean | 23.66% |
| Panopticon | S1+S2 | mean | 21.07% |
| CROMA | S2 | mean | 21.07% |
| AgriFM (geobreeze, stack) | S2 | stack T=32 | 11.09% |
| AgriFM (geobreeze, mean) | S2 | mean | ~9% |

### CropSTS published SoTA (parcel-aware): 39.09% mIoU

> Note: AgriFM standalone and CropSTS use full temporal sequences. Geobreeze experiments use frozen backbones with single-image (mean-collapsed) input except where noted. Parcel-aware models (CropSTS, UTAE) have access to field boundary information not available to foundation models.

## Results on GEO-Bench m-SA-crop-type

| Model | mIoU |
|---|---|
| Panopticon | see `geobench-m-SA-crop-type/panopticon_m_SA_results.csv` |
| CROMA | see `geobench-m-SA-crop-type/croma_m_SA_results.csv` |

## Repository Structure
```
├── agrifm.py                          # AgriFM geobreeze wrapper
├── satlaspretrain_initial_mean.py     # SatlasPretrain (mean temporal)
├── satlaspretrain_new_temporal_stack.py # SatlasPretrain (stack temporal)
├── prithvi.py                         # Prithvi-EO-2.0-600M-TL wrapper
├── ssl4eo.py                          # SSL4EO-S12 DINO wrapper
├── olmoearth.py                       # OlmoEarth wrapper
├── pastis_initial_temporal_strategy_mean.py  # PASTIS dataset (mean)
├── pastis_new_temporal_strategy_stack.py     # PASTIS dataset (stack)
├── geobench.py                        # GEO-Bench dataset
├── mmcv_mock_module.py                # mmcv mock (required for AgriFM)
├── configs/                           # Model and dataset configs
├── results/                           # Test set results CSVs
├── visualization/                     # Visualization scripts
├── geobench-m-SA-crop-type/           # GEO-Bench results and scripts
└── agrifm-pastis-outside-geobreeze/   # AgriFM standalone temporal training
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

## AgriFM Outside Geobreeze

See `agrifm-pastis-outside-geobreeze/` for full temporal training scripts achieving **45.08% mIoU** on PASTIS-R test set — outperforming published SoTA (CropSTS: 39.09%) without using parcel boundary information.

## Requirements

- `mmcv_mock_module.py` — mock for mmcv (required for AgriFM imports, zero effect on results)
- `DATASETS_DIR` environment variable pointing to dataset directory
- PASTIS-R data at `$DATASETS_DIR/pastis-r`
- GEO-Bench data at `$GEO_BENCH_DIR`
