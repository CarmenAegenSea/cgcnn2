"""Data cleaning utilities for ABX3 dataset.

Functions:
 - is_abx3(formula): detect ABX3 formulas (counts [3,1,1])
 - clean_dataframe(df, ...): perform filtering, dedup, outlier removal

CLI: run to write cleaned CSV.
"""
from __future__ import annotations
import re
import pandas as pd
import numpy as np
from typing import Optional

try:
    # Prefer package-style import when running from workspace root
    from XGBoost.featurizer import ABX3Featurizer
except Exception:
    try:
        from featurizer import ABX3Featurizer
    except Exception:
        ABX3Featurizer = None


def _parse_formula_local(formula: str) -> dict:
    """Fallback simple formula parser: returns {element: count}."""
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', str(formula))
    comp = {}
    for sym, cnt in tokens:
        cnt = int(cnt) if cnt else 1
        comp[sym] = comp.get(sym, 0) + cnt
    return comp


def is_abx3(formula: str) -> bool:
    """Return True if formula corresponds to ABX3 stoichiometry (counts [3,1,1])."""
    if not isinstance(formula, str):
        return False
    try:
        comp = ABX3Featurizer.parse_formula(formula)
        counts = [int(v) for v in comp.values()]
        # normalize by gcd to handle formulas like Ba2Ti2O6 -> counts [2,2,6] -> normalized [1,1,3]
        from math import gcd
        from functools import reduce
        g = reduce(gcd, counts)
        norm = sorted([c // g for c in counts], reverse=True)
        return norm == [3, 1, 1]
    except Exception:
        return False


def clean_dataframe(
    df: pd.DataFrame,
    target: str = 'band_gap',
    filter_mode: str = 'abx3',
    dedupe: bool = True,
    outlier_method: Optional[str] = 'zscore',
    z_thresh: float = 3.0,
    iqr_mult: float = 1.5,
) -> pd.DataFrame:
    """Clean dataframe in-place copy and return cleaned copy.

    Steps:
    - drop rows with NaN target
    - optional: keep only rows whose `formula` is ABX3
    - optional: deduplicate by `formula` (mean of numeric columns)
    - optional: remove outliers in target using z-score or IQR
    """
    df = df.copy()

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe")

    # drop missing target
    df = df.dropna(subset=[target])

    # filter according to filter_mode
    if 'formula' in df.columns and filter_mode is not None and filter_mode != 'none':
        parse_fn = None
        if ABX3Featurizer is not None:
            parse_fn = ABX3Featurizer.parse_formula
        else:
            parse_fn = _parse_formula_local

        if filter_mode == 'abx3':
            def _is_abx3_local(f):
                try:
                    comp = parse_fn(f)
                    counts = [int(v) for v in comp.values()]
                    from math import gcd
                    from functools import reduce
                    g = reduce(gcd, counts)
                    norm = sorted([c // g for c in counts], reverse=True)
                    return norm == [3, 1, 1]
                except Exception:
                    return False
            mask = df['formula'].apply(lambda f: _is_abx3_local(f) if pd.notnull(f) else False)
        elif filter_mode == 'three_elements':
            mask = df['formula'].apply(lambda f: (len(parse_fn(f)) == 3) if pd.notnull(f) else False)
        elif filter_mode == 'two_or_more':
            mask = df['formula'].apply(lambda f: (len(parse_fn(f)) >= 2) if pd.notnull(f) else False)
        else:
            mask = pd.Series([True] * len(df), index=df.index)
        df = df[mask].copy()

    # deduplicate by formula taking mean of numeric columns
    if dedupe and 'formula' in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            agg = {c: 'mean' for c in num_cols}
            df = df.groupby('formula', as_index=False).agg(agg)

    # outlier removal
    if outlier_method == 'zscore' and len(df) > 0:
        y = df[target].values
        mean = np.mean(y)
        std = np.std(y)
        if std > 0:
            z = np.abs((y - mean) / std)
            df = df[z <= z_thresh].copy()
    elif outlier_method == 'iqr' and len(df) > 0:
        q1 = df[target].quantile(0.25)
        q3 = df[target].quantile(0.75)
        iqr = q3 - q1
        low = q1 - iqr_mult * iqr
        high = q3 + iqr_mult * iqr
        df = df[(df[target] >= low) & (df[target] <= high)].copy()

    return df


def summarize(df: pd.DataFrame, target: str = 'band_gap') -> dict:
    df2 = df.dropna(subset=[target])
    y = df2[target].values
    return {
        'n': int(len(df2)),
        'mean': float(np.mean(y)) if len(y) else None,
        'median': float(np.median(y)) if len(y) else None,
        'std': float(np.std(y)) if len(y) else None,
        'min': float(np.min(y)) if len(y) else None,
        'max': float(np.max(y)) if len(y) else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare/clean ABX3 dataset')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output cleaned CSV file')
    parser.add_argument('--target', default='band_gap')
    parser.add_argument('--no-dedupe', dest='dedupe', action='store_false')
    parser.add_argument('--filter-mode', dest='filter_mode', choices=['abx3', 'three_elements', 'two_or_more', 'none'], default='abx3', help='Filtering mode to apply')
    parser.add_argument('--outlier-method', default='zscore', choices=['zscore', 'iqr', 'none'])
    parser.add_argument('--z-thresh', type=float, default=3.0)
    parser.add_argument('--iqr-mult', type=float, default=1.5)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    before = summarize(df, target=args.target)
    cleaned = clean_dataframe(
        df,
        target=args.target,
        filter_mode=args.filter_mode,
        dedupe=args.dedupe,
        outlier_method=(None if args.outlier_method == 'none' else args.outlier_method),
        z_thresh=args.z_thresh,
        iqr_mult=args.iqr_mult,
    )
    after = summarize(cleaned, target=args.target)
    cleaned.to_csv(args.output, index=False)
    print('before:', before)
    print('after:', after)


if __name__ == '__main__':
    main()
