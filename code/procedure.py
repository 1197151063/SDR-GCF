import torch

from utils import Fast_Sampling


def train(dataset, model, optimizer, edge_filter=None):
    model.train()
    total_loss = 0.0
    batches = Fast_Sampling(dataset, model.config)
    for edges, edge_ids in batches:
        optimizer.zero_grad(set_to_none=True)
        loss, instance_loss = model.get_loss(edges)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss.')
        if edge_filter is not None and edge_filter.filter_epoch is None:
            edge_filter.update(edge_ids, instance_loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
    return float(total_loss / len(batches))


@torch.no_grad()
def test(dataset, model, batch_size=512, k=20):
    model.eval()
    users, items = model()
    observed = dataset.observed_edge_index
    test_edges = dataset.test_edge_index
    k = min(k, dataset.num_items)
    discount = 1 / torch.log2(torch.arange(2, k + 2, device=users.device).float())
    ideal = torch.cat([discount.new_zeros(1), discount.cumsum(0)])
    recall, ndcg, count = 0.0, 0.0, 0

    for start in range(0, dataset.num_users, batch_size):
        end = min(start + batch_size, dataset.num_users)
        scores = users[start:end] @ items.t()
        mask = (observed[0] >= start) & (observed[0] < end)
        batch_train = observed[:, mask]
        scores[batch_train[0] - start, batch_train[1]] = float('-inf')
        mask = (test_edges[0] >= start) & (test_edges[0] < end)
        batch_test = test_edges[:, mask]
        truth = torch.zeros_like(scores, dtype=torch.bool)
        truth[batch_test[0] - start, batch_test[1]] = True
        positives = torch.bincount(batch_test[0] - start, minlength=end - start)
        valid = positives > 0
        values, indices = scores.topk(k, dim=1)
        hits = truth.gather(1, indices) & torch.isfinite(values)
        recall += float((hits.sum(1) / positives.clamp_min(1))[valid].sum())
        ndcg += float(((hits * discount).sum(1) / ideal[positives.clamp(1, k)])[valid].sum())
        count += int(valid.sum())
    if not count:
        raise ValueError('No test users to evaluate.')
    return recall / count, ndcg / count
