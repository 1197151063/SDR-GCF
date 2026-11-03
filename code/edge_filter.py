import numpy as np
import torch
from scipy.stats import rankdata


HASH_DIM = 128
PRIME = 2147483647


def mix64(value):
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def hashes(ids, salt, start=0, end=HASH_DIM):
    bases = [(salt << 32) + dim for dim in range(start, end)]
    a = torch.tensor([mix64(x) % (PRIME - 1) + 1 for x in bases], device=ids.device)
    b = torch.tensor([mix64(x ^ 0xD1B54A32D192ED03) % PRIME for x in bases], device=ids.device)
    return ((ids[:, None] * a + b) % PRIME).int()


def signatures(source, group, size, salt):
    first = torch.full((size, HASH_DIM), PRIME, dtype=torch.int32, device=source.device)
    second = torch.full_like(first, PRIME)
    for start in range(0, HASH_DIM, 8):
        values = hashes(source, salt, start, start + 8)
        index = group[:, None].expand_as(values)
        minimum = torch.full((size, 8), PRIME, dtype=torch.int32, device=source.device)
        minimum.scatter_reduce_(0, index, values, reduce='amin', include_self=True)
        matches = values == minimum[group]
        counts = torch.zeros_like(minimum).scatter_add_(0, index, matches.int())
        remaining = values.masked_fill(matches, PRIME)
        next_minimum = torch.full_like(minimum, PRIME)
        next_minimum.scatter_reduce_(0, index, remaining, reduce='amin', include_self=True)
        first[:, start:start + 8] = minimum
        second[:, start:start + 8] = torch.where(counts > 1, minimum, next_minimum)
    return first, second


def representatives(group, size, limit):
    ids = torch.arange(group.numel(), device=group.device)
    order = torch.argsort(group * PRIME + (ids * 1103515245 + 12345) % PRIME)
    sorted_groups = group[order]
    degree = torch.bincount(sorted_groups, minlength=size)
    starts = degree.cumsum(0) - degree
    offset = ids - torch.repeat_interleave(starts, degree)
    keep = offset < limit
    result = torch.full((size, limit), -1, device=group.device, dtype=torch.long)
    result[sorted_groups[keep], offset[keep]] = order[keep]
    return result


class TwoHopStructure:
    def __init__(self, edges, num_users, num_items, topk=10):
        self.original = edges
        graph = torch.sparse_coo_tensor(edges, torch.ones(edges.size(1), device=edges.device),
                                        (num_users, num_items)).coalesce()
        self.edges = graph.indices()
        users, items = self.edges
        self.user_degree = torch.bincount(users, minlength=num_users).float()
        self.item_degree = torch.bincount(items, minlength=num_items).float()
        self.item_hash = signatures(users, items, num_items, 17)
        self.user_hash = signatures(items, users, num_users, 53)
        limit = max(topk + 1, 16)
        self.user_neighbors = representatives(users, num_users, limit)
        self.item_neighbors = representatives(items, num_items, limit)

    def side(self, nodes, targets, first_second, salt, neighbors, row, degree, valid):
        first, second = first_second
        full = first[targets]
        # Remove the target edge from the candidate's neighborhood analytically.
        candidate = torch.where(full == hashes(nodes, salt), second[targets], full)
        sampled = neighbors[nodes]
        other = self.edges[row][sampled.clamp_min(0)]
        mask = (sampled >= 0) & (other != targets[:, None]) & valid[:, None]
        neighbor_degree = degree[other]
        candidate_degree = (degree[targets] - 1)[:, None]
        jaccard = (candidate[:, None] == first[other]).float().mean(dim=2)
        overlap = jaccard * (candidate_degree + neighbor_degree) / (1 + jaccard)
        overlap = torch.minimum(overlap, torch.minimum(candidate_degree, neighbor_degree))
        cosine = overlap / (candidate_degree.clamp_min(1) * neighbor_degree.clamp_min(1)).sqrt()
        values = torch.where(mask, cosine.clamp(0, 1), 0.0)
        count = mask.sum(dim=1)
        mean = values.sum(dim=1) / count.clamp_min(1)
        return mean.masked_fill(count == 0, float('nan'))

    @torch.no_grad()
    def compute(self, chunk_size=8192):
        sides = np.full((2, self.original.size(1)), np.nan, dtype=np.float64)
        for start in range(0, self.original.size(1), chunk_size):
            end = min(start + chunk_size, self.original.size(1))
            users, items = self.original[:, start:end]
            valid = (self.user_degree[users] > 1) & (self.item_degree[items] > 1)
            user_side = self.side(users, items, self.item_hash, 17, self.user_neighbors,
                                  1, self.item_degree, valid)
            item_side = self.side(items, users, self.user_hash, 53, self.item_neighbors,
                                  0, self.user_degree, valid)
            sides[:, start:end] = torch.stack([user_side, item_side]).cpu().numpy()
        finite = np.isfinite(sides)
        count = finite.sum(axis=0)
        score = np.where(finite, sides, 0).sum(axis=0) / count.clip(1)
        score[count == 0] = np.nan
        return score


def percentile(values):
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    count = int(valid.sum())
    if count:
        result[valid] = (rankdata(values[valid]) - 1) / (count - 1) if count > 1 else 0.5
    return result


class EdgeFilter:
    def __init__(self, dataset, config):
        self.config = config
        self.edges = dataset.train_edge_index
        self.num_users, self.num_items = dataset.num_users, dataset.num_items
        self.momentum = torch.zeros(self.edges.size(1), device=self.edges.device)
        self.seen = torch.zeros_like(self.momentum, dtype=torch.bool)
        self.structure = None
        self.previous = None
        self.stable_count = 0
        self.filter_epoch = None

    @torch.no_grad()
    def update(self, edge_ids, loss):
        rho = self.config['rho']
        values = rho * self.momentum[edge_ids] + (1 - rho) * loss.detach()
        self.momentum[edge_ids] = torch.where(self.seen[edge_ids], values, loss.detach())
        self.seen[edge_ids] = True

    def select(self):
        if self.structure is None:
            self.structure = TwoHopStructure(self.edges, self.num_users, self.num_items,
                                             self.config['topk']).compute(self.config['chunk_size'])
        momentum = self.momentum.detach().cpu().numpy().astype(np.float64)
        seen = self.seen.cpu().numpy() & np.isfinite(momentum)
        valid = np.isfinite(self.structure)
        retained = np.ones(self.edges.size(1), dtype=bool)
        if not valid.any() or not seen.any():
            return retained
        high_loss = seen & (momentum >= np.quantile(momentum[seen], self.config['q_loss']))
        low_structure = valid & (self.structure <= np.quantile(self.structure[valid], self.config['q_structure']))
        budget = min(int((high_loss & low_structure).sum()), int(len(retained) * self.config['cap']))
        loss_rank = np.full(len(momentum), 0.5)
        loss_rank[seen] = percentile(momentum[seen])
        weight = self.config['structure_weight']
        risk = 1 - (weight * percentile(self.structure) + (1 - weight) * (1 - loss_rank))
        eligible = np.flatnonzero(np.isfinite(risk))
        order = np.lexsort((eligible, -risk[eligible]))
        retained[eligible[order[:budget]]] = False
        if not retained.any():
            raise ValueError('Filtering would remove every training edge. Reduce --cap.')
        return retained

    def step(self, epoch):
        if self.filter_epoch is not None:
            return None
        config = self.config
        fixed = config['filter_schedule'] == 'fixed'
        if (fixed and epoch != config['filter_epoch']) or (not fixed and epoch < config['min_epoch']):
            return None
        retained = self.select()
        coverage = float(self.seen.float().mean())
        ready = fixed
        if not fixed:
            removed = ~retained
            if coverage >= config['coverage'] and self.previous is not None:
                union = np.logical_or(removed, self.previous).sum()
                similarity = np.logical_and(removed, self.previous).sum() / union if union else 1.0
                self.stable_count = self.stable_count + 1 if similarity >= config['jaccard'] else 0
            else:
                self.stable_count = 0
            self.previous = removed.copy() if coverage >= config['coverage'] else None
            ready = self.stable_count >= config['stable_checks'] or epoch >= config['max_epoch']
        if ready:
            self.filter_epoch = epoch
            return torch.from_numpy(retained).to(self.edges.device)
        return None
