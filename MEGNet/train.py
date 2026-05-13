"""
MEGNet training & evaluation script.

Trains MEGNet on the same dataset as CGCNN and reports
MAE, RMSE, R2 for direct comparison.

Usage:
  python MEGNet/train.py data/catalysis/cif [--epochs 200] [--batch-size 256]
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from random import sample, seed as set_seed

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from torch.optim.lr_scheduler import MultiStepLR

# Use EXACTLY the same data pipeline as CGCNN
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cgcnn.cgcnn.data import CIFData, collate_pool, get_train_val_test_loader

from MEGNet.model import MEGNet


warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

# ======================================================================
# Utility classes
# ======================================================================

class Normalizer:
    """Z-score normaliser (same as CGCNN)."""
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)
        if self.std < 1e-8:
            self.std = torch.tensor(1.0)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class AverageMeter:
    """Running average tracker."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ======================================================================
# Training / Validation / Test
# ======================================================================

def train_epoch(loader, model, criterion, optimizer, normalizer, args, epoch):
    model.train()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    for i, (input, target, _) in enumerate(loader):
        if args.cuda:
            input_var = tuple(x.cuda(non_blocking=True) if torch.is_tensor(x)
                              else [y.cuda(non_blocking=True) for y in x]
                              for x in input)
        else:
            input_var = input

        target_normed = normalizer.norm(target)
        if args.cuda:
            target_var = target_normed.cuda(non_blocking=True)
        else:
            target_var = target_normed

        output = model(*input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        pred = normalizer.denorm(output.data.cpu())
        mae = torch.mean(torch.abs(target - pred))
        losses.update(loss.item(), target.size(0))
        mae_errors.update(mae.item(), target.size(0))

        if i % args.print_freq == 0:
            print(f'Epoch [{epoch}][{i:3d}/{len(loader):3d}]  '
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                  f'MAE {mae_errors.val:.4f} ({mae_errors.avg:.4f})')

    return mae_errors.avg


@torch.no_grad()
def evaluate(loader, model, criterion, normalizer, args, test=False):
    model.eval()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    all_preds, all_targets, all_ids = [], [], []

    for i, (input, target, batch_ids) in enumerate(loader):
        if args.cuda:
            input_var = tuple(x.cuda(non_blocking=True) if torch.is_tensor(x)
                              else [y.cuda(non_blocking=True) for y in x]
                              for x in input)
        else:
            input_var = input

        target_normed = normalizer.norm(target)
        if args.cuda:
            target_var = target_normed.cuda(non_blocking=True)
        else:
            target_var = target_normed

        output = model(*input_var)
        loss = criterion(output, target_var)

        pred = normalizer.denorm(output.data.cpu())
        mae = torch.mean(torch.abs(target - pred))
        losses.update(loss.item(), target.size(0))
        mae_errors.update(mae.item(), target.size(0))

        if test:
            all_preds.extend(pred.view(-1).tolist())
            all_targets.extend(target.view(-1).tolist())
            all_ids.extend(batch_ids)

    if test:
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        rmse = float(np.sqrt(mean_squared_error(all_targets, all_preds)))
        r2 = float(r2_score(all_targets, all_preds))
        mae_val = float(mean_absolute_error(all_targets, all_preds))
        # Save test predictions
        test_out = os.path.join(os.path.dirname(__file__), 'test_predictions.csv')
        with open(test_out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['id', 'target', 'prediction'])
            for mid, t, p in zip(all_ids, all_targets, all_preds):
                w.writerow([mid, f'{t:.4f}', f'{p:.4f}'])
        print(f'Test predictions saved -> {test_out}')
        return {'mae': mae_val, 'rmse': rmse, 'r2': r2}

    return mae_errors.avg


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='MEGNet band-gap prediction')
    parser.add_argument('data_root', help='Path to data dir (same CGCNN format: id_prop.csv, atom_init.json, *.cif)')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train-ratio', default=0.8, type=float)
    parser.add_argument('--val-ratio', default=0.1, type=float)
    parser.add_argument('--test-ratio', default=0.1, type=float)
    parser.add_argument('--node-dim', default=64, type=int)
    parser.add_argument('--edge-dim', default=64, type=int)
    parser.add_argument('--hidden-dim', default=128, type=int)
    parser.add_argument('--n-blocks', default=3, type=int)
    parser.add_argument('--h-fea-len', default=128, type=int)
    parser.add_argument('--n-h', default=1, type=int)
    parser.add_argument('--disable-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--loss', choices=['mse', 'huber', 'l1'], default='huber')
    parser.add_argument('--print-freq', default=50, type=int)
    args = parser.parse_args()

    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    MEGNet_DIR = os.path.dirname(__file__)

    # -- Reproducibility ---------------------------------------------------
    if args.seed is not None:
        set_seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # -- Data --------------------------------------------------------------
    print('Loading dataset ...')
    dataset = CIFData(args.data_root, random_seed=args.seed if args.seed is not None else 123)

    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset, collate_fn=collate_pool, batch_size=args.batch_size,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        test_ratio=args.test_ratio, num_workers=0, return_test=True,
        train_size=None, val_size=None, test_size=None)
    print(f'Dataset size: {len(dataset)}  '
          f'Train: {len(train_loader.dataset)}  '
          f'Val: {len(val_loader.dataset)}  '
          f'Test: {len(test_loader.dataset)}')

    # Normaliser
    n_sample = min(500, len(dataset))
    sample_list = [dataset[i] for i in sample(range(len(dataset)), n_sample)]
    _, sample_target, _ = collate_pool(sample_list)
    normalizer = Normalizer(sample_target)
    print(f'Target stats - mean: {normalizer.mean:.4f}  std: {normalizer.std:.4f}')

    # -- Model -------------------------------------------------------------
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    print(f'Feature dims - atom: {orig_atom_fea_len}  nbr: {nbr_fea_len}')

    model = MEGNet(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'MEGNet created - parameters: {n_params:,}')

    if args.cuda:
        model.cuda()

    # Loss
    loss_map = {'mse': nn.MSELoss(), 'huber': nn.HuberLoss(), 'l1': nn.L1Loss()}
    criterion = loss_map[args.loss]

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)

    # -- Training -----------------------------------------------------------
    print(f'\nStarting training  ({args.epochs} epochs, {args.loss} loss)')
    print('=' * 55)
    best_mae = float('inf')
    best_epoch = -1
    train_log = []

    for epoch in range(args.start_epoch if hasattr(args, 'start_epoch') else 0,
                       args.epochs):
        t0 = time.time()
        train_mae = train_epoch(train_loader, model, criterion, optimizer,
                                normalizer, args, epoch + 1)
        val_mae = evaluate(val_loader, model, criterion, normalizer, args,
                           test=False)

        elapsed = time.time() - t0
        train_log.append({'epoch': epoch + 1, 'train_mae': train_mae,
                          'val_mae': val_mae, 'time': f'{elapsed:.1f}s'})

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'  [{epoch+1:3d}/{args.epochs}]  '
                  f'Train MAE: {train_mae:.4f}  Val MAE: {val_mae:.4f}  '
                  f'({elapsed:.1f}s)')

        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch + 1
            torch.save(model.state_dict(),
                       os.path.join(MEGNet_DIR, 'best_model.pth.tar'))
            # Also save normalizer
            torch.save(normalizer.state_dict(),
                       os.path.join(MEGNet_DIR, 'normalizer.pth'))

        scheduler.step()

    print(f'\nTraining done. Best val MAE = {best_mae:.4f} (epoch {best_epoch})')

    # -- Test evaluation ----------------------------------------------------
    print('\nEvaluating on test set ...')
    best_path = os.path.join(MEGNet_DIR, 'best_model.pth.tar')
    model.load_state_dict(torch.load(best_path, map_location='cpu'))
    if args.cuda:
        model.cuda()
    model.eval()

    metrics = evaluate(test_loader, model, criterion, normalizer, args,
                       test=True)

    print()
    print('=' * 50)
    print('  MEGNet  Performance  (vs  CGCNN  on  same  dataset)')
    print('=' * 50)
    print(f'  MAE  = {metrics["mae"]:.4f}  eV')
    print(f'  RMSE = {metrics["rmse"]:.4f}  eV')
    print(f'  R2   = {metrics["r2"]:.4f}')
    print('=' * 50)

    # -- Save metrics JSON --------------------------------------------------
    metrics['best_val_mae'] = best_mae
    metrics['best_epoch'] = best_epoch
    metrics['n_params'] = n_params
    metrics['args'] = vars(args)

    metrics_path = os.path.join(MEGNet_DIR, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Metrics saved  -> {metrics_path}')

    # Save training log
    log_path = os.path.join(MEGNet_DIR, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'train_mae', 'val_mae', 'time'])
        w.writeheader()
        w.writerows(train_log)
    print(f'Training log   -> {log_path}')


if __name__ == '__main__':
    main()
