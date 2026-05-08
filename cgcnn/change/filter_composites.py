"""
筛选复合光催化材料，提供函数接口供脚本调用。

函数 filter_and_save(pred_csv, output_dir, gap_min=1.6, gap_max=2.8)
会将筛选结果写到 output_dir/final_composites.csv 并返回 (filtered_df, out_path)

直接作为脚本使用：python cgcnn/change/filter_composites.py <pred_csv> --out <out_dir>
"""

import os
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from typing import Tuple, Optional


def _find_id_col(columns):
    cols_lower = [str(c).strip().lower() for c in columns]
    for want in ('composite_id', 'material_id', 'material id', 'id'):
        if want in cols_lower:
            return list(columns)[cols_lower.index(want)]
    return None


def _find_bandgap_col(columns):
    for c in columns:
        s = str(c).lower()
        if 'predicted' in s and 'bandgap' in s:
            return c
    for c in columns:
        s = str(c).lower()
        if s == 'bandgap' or s == 'predicted_bandgap':
            return c
    return None


def filter_and_save(pred_csv: str, output_dir: str,
                    gap_min: float = 1.6, gap_max: float = 2.8) -> Tuple[pd.DataFrame, str]:
    """Filter composite photocatalyst predictions and save final candidates to output_dir.

    Returns (filtered_df, out_csv_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    pred_df = pd.read_csv(pred_csv)

    if 'composite_id' not in pred_df.columns:
        idcol = _find_id_col(pred_df.columns)
        if idcol and idcol != 'composite_id':
            pred_df = pred_df.rename(columns={idcol: 'composite_id'})

    bg_col = _find_bandgap_col(pred_df.columns)
    if bg_col is None:
        for c in pred_df.columns:
            if 'predicted' in str(c).lower():
                bg_col = c
                break

    if bg_col is None:
        raise ValueError('未找到带隙预测列')

    cond_gap = pred_df[bg_col].between(gap_min, gap_max)

    filtered = pred_df[cond_gap].copy()

    preferred = ['composite_id', 'material1_formula', 'material2_formula',
                 'material1_bandgap', 'material2_bandgap', bg_col]
    out_cols = [c for c in preferred if c in filtered.columns]
    if 'prediction_std' in filtered.columns and 'prediction_std' in out_cols:
        pass
    elif 'prediction_std' in filtered.columns:
        out_cols.append('prediction_std')

    out_path = os.path.join(output_dir, 'final_composites.csv')
    filtered.to_csv(out_path, columns=out_cols, index=False)

    return filtered, out_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Filter composite photocatalyst materials from prediction CSV')
    parser.add_argument('pred_csv', help='predictions CSV (composite_id, predicted_bandgap, etc.)')
    parser.add_argument('--out', '-o', help='output directory to save final_composites.csv (default: ./)', default='.')
    parser.add_argument('--gap-min', type=float, default=1.6)
    parser.add_argument('--gap-max', type=float, default=2.8)
    args = parser.parse_args()

    filtered_df, out_csv = filter_and_save(args.pred_csv, args.out,
                                           gap_min=args.gap_min, gap_max=args.gap_max)
    print(f'Filtered {len(filtered_df)} composites, saved to: {out_csv}')