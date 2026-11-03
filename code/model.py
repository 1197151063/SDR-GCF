import torch
import torch.nn.functional as F
from torch import nn
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class SDRGCF(nn.Module):
    def __init__(self, num_users, num_items, config, edge_index):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.config = config
        self.user_embedding = nn.Embedding(num_users, config['dim'])
        self.item_embedding = nn.Embedding(num_items, config['dim'])
        for embedding in (self.user_embedding, self.item_embedding):
            if config['init'] == 'normal':
                nn.init.normal_(embedding.weight, std=config['init_weight'])
            else:
                nn.init.xavier_uniform_(embedding.weight, gain=config['init_weight'])
        self.set_training_graph(edge_index)

    @torch.no_grad()
    def set_training_graph(self, edge_index):
        if edge_index.size(1) == 0:
            raise ValueError('The training graph cannot be empty.')
        users, items = edge_index
        row = torch.cat([users, items + self.num_users])
        col = torch.cat([items + self.num_users, users])
        size = self.num_users + self.num_items
        graph = SparseTensor(row=row, col=col, sparse_sizes=(size, size))
        self.edge_index = gcn_norm(graph)

    def cross_norm(self, x):
        users, items = x.split([self.num_users, self.num_items])
        user_rms = (1e-6 + users.square().sum(dim=1).mean()).sqrt()
        item_rms = (1e-6 + items.square().sum(dim=1).mean()).sqrt()
        return torch.cat([users / item_rms, items / user_rms])

    def forward(self, return_layers=False):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        layers = [x]
        for _ in range(self.config['K']):
            x = matmul(self.edge_index, x)
            x = F.dropout(x, p=self.config['dropout'], training=self.training)
            mu = self.config['mu']
            if mu:
                x = (1 - mu) * x + mu * self.cross_norm(x)
            layers.append(x)
        layers = torch.stack(layers, dim=1)
        if return_layers:
            return layers
        return layers.mean(dim=1).split([self.num_users, self.num_items])

    def get_loss(self, edges):
        layers = self.forward(return_layers=True)
        users = layers[:self.num_users][edges[0]]
        items = layers[self.num_users:][edges[1]]
        u, i = users.mean(dim=1), items.mean(dim=1)
        objective = self.config['loss']

        if objective == 'bpr':
            negative = layers[self.num_users:][edges[2]].mean(dim=1)
            instance_loss = F.softplus((u * negative).sum(-1) - (u * i).sum(-1))
            ego_u = self.user_embedding(edges[0])
            ego_i = self.item_embedding(edges[1])
            ego_j = self.item_embedding(edges[2])
            reg = (ego_u.square().sum() + ego_i.square().sum() + ego_j.square().sum())
            loss = instance_loss.mean() + self.config['decay'] * reg / (2 * u.size(0))

        elif objective == 'ssm':
            if u.size(0) < 2:
                raise ValueError('SSM needs at least two interactions in a batch.')
            scores = F.normalize(u, dim=-1) @ F.normalize(i, dim=-1).t()
            scores = scores / self.config['tau']
            positive = scores.diagonal()
            diagonal = torch.eye(u.size(0), dtype=torch.bool, device=u.device)
            negative = scores.masked_fill(diagonal, float('-inf'))
            instance_loss = torch.logsumexp(negative, dim=1) - positive
            reg = users.square().sum() + items.square().sum()
            loss = instance_loss.mean() + self.config['decay'] * reg / (2 * u.size(0))

        else:
            u, i = F.normalize(u, dim=-1), F.normalize(i, dim=-1)
            instance_loss = (u - i).square().sum(dim=-1)
            uniformity = self.uniformity(u) + self.uniformity(i)
            loss = instance_loss.mean() + self.config['uniformity'] * uniformity

        # AU contributes alignment only; batch uniformity never enters the edge EMA.
        return loss, instance_loss.detach()

    def uniformity(self, x):
        if x.size(0) < 2:
            return x.sum() * 0
        return torch.pdist(x).square().mul(-self.config['uniformity_t']).exp().mean().log()
