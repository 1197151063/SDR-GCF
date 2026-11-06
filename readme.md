# Official Implementation of SDR-GCF

This repository provides the official PyTorch implementation of SDR-GCF.

## Requirements

- torch >= 2.0
- torch-geometric >= 2.5
- torch-sparse >= 0.6.17
- numpy >= 1.24
- scipy >= 1.10

Install `torch-sparse` using a wheel compatible with your PyTorch and CUDA versions.

## Reproduce Paper Results

Run the following commands from the `code/` directory:

```bash
cd code
```

Unless overridden, all experiments use 64-dimensional embeddings, 3 propagation layers, a batch size of 2,048, and random seed 2026. Models are trained for up to 500 epochs with an early-stopping patience of 20 epochs.

### SSM Loss

The following SSM configurations enable CrossNorm and disable graph reconstruction (`--filter 0`).

**Yelp2018**

```bash
python SDR-GCF.py --dataset yelp2018 --loss ssm \
    --mu 1e-3 --tau 0.14 --filter 0 --structure_weight 0.6 \
    --decay 0 --lr 1e-3 --dropout 0
```

**Amazon-Book**

```bash
python SDR-GCF.py --dataset amazon-book --loss ssm \
    --mu 1e-2 --tau 0.095 --filter 0 --init_weight 5 \
    --decay 0 --lr 1e-3 --dropout 0
```

### BPR Loss

**Yelp2018**

```bash
python SDR-GCF.py --dataset yelp2018 --loss bpr \
    --mu 0.4 --structure_weight 0.6 \
    --decay 1e-4 --lr 5e-4 --dropout 0
```

**Amazon-Book**

```bash
python SDR-GCF.py --dataset amazon-book --loss bpr \
    --mu 0.4 --structure_weight 0.6 \
    --decay 1e-4 --lr 5e-4 --dropout 0
```

### AU Loss

**Yelp2018**

```bash
python SDR-GCF.py --dataset yelp2018 --loss au \
    --mu 1.0 --structure_weight 0.6 --uniformity 1.0 \
    --decay 0 --lr 1e-3 --dropout 0
```

**Amazon-Book**

```bash
python SDR-GCF.py --dataset amazon-book --loss au \
    --mu 0.2 --structure_weight 0.6 --uniformity 5.0 \
    --decay 0 --lr 1e-3 --dropout 0
```

## Training Logs

We also provide training logs for reference.

