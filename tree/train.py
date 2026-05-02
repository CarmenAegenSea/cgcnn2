"""Train a Random Forest baseline on a CIF dataset.

Usage example:
  python tree/train.py data/catalysis/cif --out models/tree_seed42 --n-estimators 200

The script saves a model bundle (`model.joblib`) and writes test predictions
to `test_results_final.csv` under the output directory (and a copy to
`<datapath>/test_results_rf.csv`).
"""
import os
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from tree.data import build_features
from tree.model import train_random_forest, evaluate_model, predict_with_uncertainty, save_model_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', help='Path to dataset root (contains id_prop.csv and .cif files)')
    parser.add_argument('--out', default='tree/output', help='Output directory to save model and results (basename; final dir is placed under tree/)')
    parser.add_argument('--target-name', default='band_gap',
                        help='Target name to use in outputs (e.g., band_gap)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of samples to use (for smoke testing). Uses first N ids from id_prop.csv')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.1, help='Fraction used as test set')
    parser.add_argument('--val-size', type=float, default=0.1, help='Fraction used as validation set (from full data)')
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--stable-threshold', type=float, default=0.2,
                        help='Std threshold for stable predictions')
    args = parser.parse_args()

    # Ensure outputs are placed under the repository's tree/ directory
    repo_tree_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    out_dir = os.path.join(repo_tree_dir, os.path.basename(os.path.normpath(args.out)))
    os.makedirs(out_dir, exist_ok=True)

    print('Building features...')
    ids_arg = None
    if args.max_samples and args.max_samples > 0:
        id_prop_path = os.path.join(args.datapath, 'id_prop.csv')
        if os.path.exists(id_prop_path):
            df_ids = pd.read_csv(id_prop_path, header=None, names=['id', 'target'])
            ids_list = df_ids['id'].astype(str).tolist()
            ids_arg = ids_list[:args.max_samples]
            print(f'Limiting to first {len(ids_arg)} samples for smoke test.')
        else:
            print(f'Warning: id_prop.csv not found at {id_prop_path}, ignoring --max-samples')

    X, y, ids, elements = build_features(args.datapath, ids=ids_arg, verbose=True)
    print(f"Assuming target value from id_prop.csv; output target name: {args.target_name}")
    if X.shape[0] == 0:
        raise RuntimeError('No samples found when building features')

    # split out test set first
    X_tmp, X_test, y_tmp, y_test, ids_tmp, ids_test = train_test_split(
        X, y, ids, test_size=args.test_size, random_state=args.seed)

    # split validation from remaining (val_size is fraction of full data)
    val_fraction = 0.0
    if args.val_size and args.val_size > 0:
        val_fraction = args.val_size / (1.0 - args.test_size)

    if val_fraction > 0:
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_tmp, y_tmp, ids_tmp, test_size=val_fraction, random_state=args.seed)
    else:
        X_train, y_train, ids_train = X_tmp, y_tmp, ids_tmp
        X_val = y_val = ids_val = None

    print(f'Train / val / test sizes: {len(ids_train)} / {len(ids_val) if X_val is not None else 0} / {len(ids_test)}')

    print('Training RandomForest...')
    scaler, model = train_random_forest(X_train, y_train,
                                        n_estimators=args.n_estimators,
                                        max_depth=args.max_depth,
                                        random_state=args.seed,
                                        n_jobs=args.n_jobs)

    # evaluate
    if X_val is not None:
        val_metrics = evaluate_model(scaler, model, X_val, y_val)
        print('Validation metrics:', val_metrics)

    test_metrics = evaluate_model(scaler, model, X_test, y_test)
    print('Test metrics:', test_metrics)

    # predictions with uncertainty
    preds_mean, preds_std = predict_with_uncertainty(scaler, model, X_test)

    # write CSV in out dir
    out_csv = os.path.join(out_dir, 'test_results_final.csv')
    import csv
    header = ['id', args.target_name, 'prediction', 'prediction_std']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, mid in enumerate(ids_test):
            writer.writerow([mid, f"{float(y_test[i]):.6f}", f"{float(preds_mean[i]):.6f}", f"{float(preds_std[i]):.6f}"])


    # save model bundle
    model_path = os.path.join(out_dir, 'model.joblib')
    save_model_bundle(model_path, scaler, model, elements)
    print('Saved model bundle to', model_path)

    # save metadata
    meta = {'n_samples': int(X.shape[0]), 'n_elements': len(elements), 'elements': elements, 'test_metrics': test_metrics, 'target_name': args.target_name}
    with open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print('Training complete. Predictions saved to', out_csv)


if __name__ == '__main__':
    main()
