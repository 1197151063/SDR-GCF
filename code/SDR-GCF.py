import json
import logging
import time

import torch

from dataloader import Loader
from edge_filter import EdgeFilter
from model import SDRGCF
from parse import parse_args
from procedure import test, train
from utils import init_logger, set_seed, sync_device


def main():
    config = parse_args()
    set_seed(config['seed'])
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if config['device'] == 'auto' else config['device'])
    stem = init_logger(config)
    logging.info(json.dumps(config, indent=2))
    dataset = Loader(config, device)
    if config['loss'] == 'ssm' and dataset.train_edge_index.size(1) < 2:
        raise ValueError('SSM needs at least two training interactions.')
    model = SDRGCF(dataset.num_users, dataset.num_items, config,
                   dataset.train_edge_index).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    edge_filter = EdgeFilter(dataset, config) if config['filter'] else None
    best, best_post_filter = None, None
    patience, filtering, history = 0, None, []
    total_time = 0.0

    for epoch in range(1, config['epochs'] + 1):
        sync_device(device)
        start = time.perf_counter()
        loss = train(dataset, model, optimizer, edge_filter)
        retained = edge_filter.step(epoch) if edge_filter is not None else None
        if retained is not None:
            dataset.filter_edges(retained)
            model.set_training_graph(dataset.train_edge_index)
            filtering = dict(epoch=epoch, removed=int((~retained).sum()), retained=int(retained.sum()))
            logging.info('Filtering: %s', filtering)
        sync_device(device)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        recall, ndcg = test(dataset, model, config['testbatch'])
        result = dict(epoch=epoch, recall_at_20=recall, ndcg_at_20=ndcg, training_seconds=total_time)
        if best is None or recall > best['recall_at_20']:
            best = result
            patience = 0
        else:
            patience += 1
        if filtering is not None and (best_post_filter is None or recall > best_post_filter['recall_at_20']):
            best_post_filter = result
        history.append(dict(**result, loss=loss, epoch_seconds=elapsed))
        logging.info('Epoch: %03d, loss: %.5f, R@20: %.6f, N@20: %.6f, time: %.2f seconds',
                     epoch, loss, recall, ndcg, elapsed)
        if patience >= config['patience']:
            break

    summary = dict(config=config, best=best, best_post_filter=best_post_filter,
                   filtering=filtering, epochs=len(history), training_seconds=total_time,
                   seconds_per_epoch=total_time / len(history), history=history)
    stem.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logging.info('\n========== BEST RESULT ==========')
    logging.info('Best Epoch: %d', best['epoch'])
    logging.info('Recall@20: %.6f, NDCG@20: %.6f', best['recall_at_20'], best['ndcg_at_20'])
    logging.info('Log saved to: %s.log', stem)


if __name__ == '__main__':
    main()
