# Geobreeze Crop Mapping Extensions

This repository contains extensions to [geobreeze](https://github.com/geobreeze/geobreeze) for crop type mapping evaluation on the PASTIS-R and GEO-Bench m-SA-crop-type datasets.

## New Models Integrated into Geobreeze

| Model | File | Config |
|---|---|---|
| AgriFM | `agrifm.py` | `configs/agrifm.yaml` |
| SatlasPretrain | `satlaspretrain.py` | `configs/satlaspretrain.yaml` |
| Prithvi-EO-2.0-600M-TL | `prithvi.py` | `configs/prithvi_600m_tl.yaml` |

## Results on PASTIS-R Test Set (mIoU)

| Model | Temporal | mIoU |
|---|---|---|
| AgriFM temporal (outside geobreeze) | T=32 | 45.08% |
| Panopticon S1+S2 | mean | 21.07% |
| CROMA S2 | mean | 21.07% |
| SatlasPretrain S2 | mean | 23.66% |
| AgriFM single-image | mean | ~9% |

## Usage

### Install geobreeze
```bash
git clone https://github.com/geobreeze/geobreeze
cd geobreeze
pip install -e .
```

### Copy model files
```bash
cp agrifm.py satlaspretrain.py prithvi.py geobreeze/geobreeze/models/
cp configs/*.yaml geobreeze/geobreeze/config/model/base/
cp pastis.py geobench.py geobreeze/geobreeze/datasets/
```

### Run experiments
```bash
# PASTIS with Panopticon
python geobreeze/main.py +model=base/panopticon +data=pastis-all +optim=segmentation output_dir=outputs/panopticon_pastis

# PASTIS with SatlasPretrain
python geobreeze/main.py +model=base/satlaspretrain +data=pastis-s2 +optim=segmentation output_dir=outputs/satlaspretrain_pastis

# GEO-Bench m-SA-crop-type with Panopticon
python geobreeze/main.py +model=base/panopticon +data=m-SA-crop-type +optim=segmentation output_dir=outputs/panopticon_SA
```

## AgriFM Outside Geobreeze
See `agrifm-pastis-outside-geobreeze/` for full temporal training scripts.

## Requirements
- `mmcv_mock_module.py` — mock for mmcv (required for AgriFM imports)
- GEO_BENCH_DIR must point to geobench dataset directory
- PASTIS-R data at DATASETS_DIR/pastis-r
