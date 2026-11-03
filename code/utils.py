import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Fast_Sampling(dataset, config):
    edges = dataset.train_edge_index
    batch_size = config['batch_size']
    if config['loss'] == 'bpr':
        order = DataLoader(range(edges.size(1)), batch_size=batch_size, shuffle=True)
    else:
        indices = np.arange(edges.size(1))
        np.random.shuffle(indices)
        order = []
        start = 0
        while start < len(indices):
            remaining = len(indices) - start
            take = batch_size + 1 if remaining == batch_size + 1 else min(batch_size, remaining)
            order.append(torch.from_numpy(indices[start:start + take]))
            start += take
    batches = []
    for index in order:
        index = index.to(edges.device)
        positives = edges[:, index]
        if config['loss'] == 'bpr':
            negative = torch.randint(dataset.num_items, (index.numel(),), device=edges.device)
            positives = torch.cat([positives, negative.unsqueeze(0)])
        batches.append((positives, dataset.edge_ids[index]))
    return batches


def init_logger(config):
    directory = Path(config['log_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    stem = directory / f"SDR-GCF_{config['dataset']}_{config['loss']}_{stamp}"
    logging.basicConfig(level=logging.INFO, format='%(message)s', force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(stem.with_suffix('.log'))])
    return stem


def sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
