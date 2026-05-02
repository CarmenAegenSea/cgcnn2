"""Predict with a saved RandomForest model bundle.

Usage:
  python tree/predict.py models/tree/model.joblib data/catalysis/cif --csv-output out.csv
"""
import os
import argparse
import joblib
import csv

from tree.data import build_features
from tree.model import predict_with_uncertainty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modelpath', help='Path to model.joblib saved by train.py')
    parser.add_argument('datapath', help='Path to dataset root (contains id_prop.csv and .cif files)')
    parser.add_argument('--target-name', default='band_gap',
                        help='Target name to use in CSV output (e.g., band_gap)')
    parser.add_argument('--csv-output', default=None, help='Optional CSV output path (must be inside tree/; otherwise basename is used)')
    parser.add_argument('--ensemble-threshold', type=float, default=0.2,
                        help='Std threshold below which a prediction is considered stable')
    args = parser.parse_args()

    bundle = joblib.load(args.modelpath)
    model = bundle['model']
    scaler = bundle['scaler']
    elements = bundle.get('elements', None)

    # build features using training element ordering (if provided)
    X, y, ids, _ = build_features(args.datapath, element_list=elements, verbose=False)
    if X.shape[0] == 0:
        raise RuntimeError('No samples found when building features')

    mean_preds, std_preds = predict_with_uncertainty(scaler, model, X)

    # Determine output CSV path and ensure it is inside the repository's tree/ directory
    tree_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    if args.csv_output:
        candidate = os.path.abspath(args.csv_output)
        try:
            # if candidate is not under tree_dir, place file inside tree_dir using basename
            if os.path.commonpath([candidate, tree_dir]) != tree_dir:
                out_file = os.path.join(tree_dir, os.path.basename(candidate))
            else:
                out_file = candidate
        except Exception:
            out_file = os.path.join(tree_dir, os.path.basename(candidate))
    else:
        modelbase = os.path.splitext(os.path.basename(args.modelpath))[0]
        out_file = os.path.join(tree_dir, f'test_results_rf_{modelbase}.csv')

    header = ['id', args.target_name, 'prediction', 'prediction_std']
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, mid in enumerate(ids):
            tgt = float(y[i]) if y is not None and len(y) > i else ''
            writer.writerow([mid, f"{tgt:.6f}" if tgt != '' else '', f"{float(mean_preds[i]):.6f}", f"{float(std_preds[i]):.6f}"])

    print('Predictions saved to', out_file)


if __name__ == '__main__':
    main()
