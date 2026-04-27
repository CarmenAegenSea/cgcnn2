#!/usr/bin/env python3
"""Train multiple models with different random seeds and save checkpoints per-seed.

Usage example:
  python cgcnn/train_ensemble.py data/catalysis/cif --seeds 42 2021 7 --out models/ensemble --extra-args "--epochs 200 --batch-size 32 --lr 0.01"

This script runs `cgcnn/main.py` in separate working directories so each run produces its own `model_best.pth.tar`.
"""

import os
import sys
import subprocess
import argparse
import shlex


def main():
    parser = argparse.ArgumentParser(description='Train ensemble models with different seeds')
    parser.add_argument('data_root', help='Path to dataset root (passed to cgcnn/main.py)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 2021, 7],
                        help='List of integer seeds to train (default: 42 2021 7)')
    parser.add_argument('--out', default='models/ensemble', help='Output base directory to store per-seed folders')
    parser.add_argument('--extra-args', default='', help='Extra args string forwarded to cgcnn/main.py (quote as needed)')
    parser.add_argument('--disable-cuda', action='store_true', help='Pass --disable-cuda to training runs')
    args = parser.parse_args()

    main_py = os.path.join(os.path.dirname(__file__), 'main.py')
    os.makedirs(args.out, exist_ok=True)

    for seed in args.seeds:
        outdir = os.path.join(args.out, f'seed_{seed}')
        os.makedirs(outdir, exist_ok=True)
        cmd = [sys.executable, main_py, '--seed', str(seed)]
        if args.disable_cuda:
            cmd.append('--disable-cuda')
        if args.extra_args:
            cmd.extend(shlex.split(args.extra_args))
        # data_root is positional arg for main.py
        cmd.append(os.path.abspath(args.data_root))

        print('--' * 30)
        print('Running training with seed=', seed)
        print('Command:', ' '.join(cmd))
        print('Work dir:', outdir)
        try:
            subprocess.run(cmd, check=True, cwd=outdir)
        except subprocess.CalledProcessError as e:
            print(f'Training failed for seed {seed}:', e)
            continue

        model_path = os.path.join(outdir, 'model_best.pth.tar')
        if os.path.exists(model_path):
            print('Saved model:', model_path)
        else:
            print('Warning: model_best.pth.tar not found in', outdir)

    print('\nEnsemble training finished. Models saved under:', os.path.abspath(args.out))


if __name__ == '__main__':
    main()
