from pathlib import Path

import torch


def read_edges(path):
    edges = []
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            values = list(map(int, line.split()))
            if values:
                edges.extend((values[0], item) for item in values[1:])
    if not edges:
        raise ValueError(f'No interactions in {path}')
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if edges.min() < 0:
        raise ValueError(f'Negative user/item ID in {path}')
    return edges


class Loader:
    def __init__(self, config, device):
        directory = Path(config['data_dir']) / config['dataset']
        train = read_edges(config['train_file'] or directory / 'train.txt')
        test = read_edges(config['test_file'] or directory / 'test.txt')
        self.num_users = int(max(train[0].max(), test[0].max())) + 1
        self.num_items = int(max(train[1].max(), test[1].max())) + 1
        self.train_edge_index = train.to(device)
        self.test_edge_index = test.to(device)
        self.observed_edge_index = self.train_edge_index
        self.edge_ids = torch.arange(train.size(1), device=device)
        print(f"{config['dataset']}: {self.num_users} users, {self.num_items} items, "
              f'{train.size(1)} train, {test.size(1)} test')

    def filter_edges(self, retained):
        self.train_edge_index = self.train_edge_index[:, retained]
        self.edge_ids = self.edge_ids[retained]
