"""
筛选候选材料，提供函数接口供脚本调用。

函数 filter_and_save(pred_csv, output_dir, attr_file=None, gap_min=1.6, gap_max=2.8)
会将筛选结果写到 output_dir/final_candidates.csv并返回(filtered_df, out_path)

可直接作为脚本使用：
    python cgcnn/change/filter_candidates.py <pred_csv> --out <out_dir>
"""

import os
import pandas as pd
from typing import Tuple, Optional


def _find_id_col(columns):
    cols_lower = [str(c).strip().lower() for c in columns]
    for want in ('material_id', 'material id', 'materialid', 'mpid', 'mp-id', 'id'):
        if want in cols_lower:
            return list(columns)[cols_lower.index(want)]
    return None


def _find_formation_energy_col(columns):
    for c in columns:
        s = str(c).lower()
        if 'formation' in s and 'energy' in s:
            return c
    return None


def filter_and_save(pred_csv: str, output_dir: str, attr_file: Optional[str] = None,
                    gap_min: float = 1.6, gap_max: float = 2.8) -> Tuple[pd.DataFrame, str]:
    """Filter predictions and save final candidates to output_dir.

    Returns (filtered_df, out_csv_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    pred_df = pd.read_csv(pred_csv)
    # normalize id column name
    if 'material_id' not in pred_df.columns:
        if 'id' in pred_df.columns:
            pred_df = pred_df.rename(columns={'id': 'material_id'})
        else:
            idcol = _find_id_col(pred_df.columns)
            if idcol and idcol != 'material_id':
                pred_df = pred_df.rename(columns={idcol: 'material_id'})

    # attr file default (repo-root/data/tmc_data/tmc_all_materials.csv)
    if attr_file is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        attr_file = os.path.join(repo_root, 'data', 'tmc_data', 'tmc_all_materials.csv')

    if not os.path.exists(attr_file):
        raise FileNotFoundError(f"属性文件未找到: {attr_file}")

    attr_df = pd.read_csv(attr_file)
    # normalize attr id column
    if 'material_id' not in attr_df.columns:
        aid = _find_id_col(attr_df.columns)
        if aid and aid != 'material_id':
            attr_df = attr_df.rename(columns={aid: 'material_id'})

    merged = pred_df.merge(attr_df, on='material_id', how='inner')

    # find formation energy column
    form_e_col = _find_formation_energy_col(merged.columns)
    if form_e_col is None:
        raise ValueError('未找到形成能列，请在属性表中包含名称含 "formation" 和 "energy" 的列')

    # build conditions (graceful if columns missing)
    cond_gap = merged['prediction'].between(gap_min, gap_max)
    cond_form = merged[form_e_col] <= 0.0 if form_e_col in merged.columns else pd.Series([True] * len(merged))
    cond_stable = merged['is_stable'] == True if 'is_stable' in merged.columns else pd.Series([True] * len(merged))

    filtered = merged[cond_gap & cond_form & cond_stable]

    # choose columns to output
    preferred = ['material_id', 'formula', 'prediction', form_e_col, 'crystal_system']
    out_cols = [c for c in preferred if c in filtered.columns]
    out_path = os.path.join(output_dir, 'final_candidates.csv')
    # always write a CSV (possibly empty) for reproducibility
    filtered.to_csv(out_path, columns=out_cols, index=False)

    return filtered, out_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Filter candidate materials from prediction CSV')
    parser.add_argument('pred_csv', help='predictions CSV (id/ material_id, target, prediction)')
    parser.add_argument('--out', '-o', help='output directory to save final_candidates.csv (default: ./)', default='.')
    parser.add_argument('--attr', help='path to attribute CSV (optional)')
    parser.add_argument('--gap-min', type=float, default=1.6)
    parser.add_argument('--gap-max', type=float, default=2.8)
    args = parser.parse_args()

    filtered_df, out_csv = filter_and_save(args.pred_csv, args.out, attr_file=args.attr,
                                          gap_min=args.gap_min, gap_max=args.gap_max)
    print(f'Filtered {len(filtered_df)} candidates, saved to: {out_csv}')