import argparse
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('lastfm', 'ml-1m', 'yelp2018', 'amazon-book')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='SDR-GCF')
    parser.add_argument('--dataset', choices=DATASETS, default='yelp2018')
    parser.add_argument('--loss', '--training-objective', choices=('bpr', 'ssm', 'au'), default='bpr')
    parser.add_argument('--data_dir', default=str(ROOT / 'data'))
    parser.add_argument('--train_file', default=None)
    parser.add_argument('--test_file', default=None)
    parser.add_argument('--log_dir', default=str(ROOT / 'log'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--bpr_batch', '--batch_size', dest='batch_size', type=int, default=2048)
    parser.add_argument('--testbatch', type=int, default=512)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--decay', type=float, default=None)
    parser.add_argument('--init', choices=('normal', 'xavier'), default=None)
    parser.add_argument('--init_weight', type=float, default=None)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--uniformity', type=float, default=None)
    parser.add_argument('--uniformity_t', type=float, default=2.0)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--mu', '--lambda_', type=float, default=None)
    parser.add_argument('--filter', type=int, choices=(0, 1), default=1)
    parser.add_argument('--structure_weight', type=float, default=None)
    parser.add_argument('--cap', type=float, default=None)
    parser.add_argument('--filter_schedule', choices=('fixed', 'adaptive'), default=None)
    parser.add_argument('--filter_epoch', type=int, default=None)
    parser.add_argument('--min_epoch', type=int, default=None)
    parser.add_argument('--max_epoch', type=int, default=None)
    parser.add_argument('--stable_checks', type=int, default=None)
    parser.add_argument('--jaccard', type=float, default=0.9)
    parser.add_argument('--coverage', type=float, default=0.99)
    parser.add_argument('--rho', type=float, default=0.9)
    parser.add_argument('--q_loss', type=float, default=0.8)
    parser.add_argument('--q_structure', type=float, default=0.2)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--chunk_size', type=int, default=8192)
    args = parser.parse_args(argv)
    config = vars(args)

    small = args.dataset in ('lastfm', 'ml-1m')
    defaults = dict(decay=1e-3 if small else 1e-4, mu=0.2 if small else 0.4,
                    structure_weight=0.95 if small else 0.6,
                    cap={'lastfm': 0.04, 'ml-1m': 0.005}.get(args.dataset, 1.0),
                    filter_schedule='fixed' if args.dataset == 'lastfm' else 'adaptive',
                    filter_epoch=10 if small else 3, min_epoch=5 if small else 2,
                    max_epoch=10 if small else 4, stable_checks=2 if small else 1,
                    init='normal' if args.loss == 'bpr' else 'xavier',
                    init_weight=0.01 if args.loss == 'bpr' else 1.0,
                    tau={'lastfm': 0.5, 'ml-1m': 0.1, 'yelp2018': 0.14, 'amazon-book': 0.1}[args.dataset],
                    uniformity={'lastfm': 0.1, 'ml-1m': 0.5, 'yelp2018': 1.0, 'amazon-book': 5.0}[args.dataset],
                    dropout=0.1 if args.loss == 'ssm' and args.dataset == 'yelp2018' else 0.0)
    if args.loss == 'au':
        defaults['mu'] = 1.0 if args.dataset in ('yelp2018', 'ml-1m') else 0.2
    for key, value in defaults.items():
        if config[key] is None:
            config[key] = value
    if config['lr'] is None:
        config['lr'] = 5e-4 if config['mu'] else 1e-3

    for key in ('epochs', 'patience', 'batch_size', 'testbatch', 'dim', 'topk', 'chunk_size', 'stable_checks', 'filter_epoch'):
        if config[key] < 1:
            parser.error(f'--{key} must be positive')
    if args.K < 0 or args.seed < 0 or args.batch_size < (2 if args.loss == 'ssm' else 1):
        parser.error('Invalid K, seed, or batch size')
    if args.min_epoch < 2 or args.max_epoch < args.min_epoch:
        parser.error('Require 2 <= min_epoch <= max_epoch')
    for key in ('mu', 'structure_weight', 'cap', 'q_loss', 'q_structure', 'jaccard', 'coverage', 'rho', 'dropout'):
        value = config[key]
        if not math.isfinite(value) or not 0 <= value <= 1 or (key in ('rho', 'dropout') and value == 1):
            parser.error(f'Invalid --{key}')
    for key in ('lr', 'tau', 'uniformity_t', 'init_weight'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            parser.error(f'--{key} must be finite and positive')
    for key in ('decay', 'uniformity'):
        if not math.isfinite(config[key]) or config[key] < 0:
            parser.error(f'--{key} must be finite and non-negative')
    return config
