"""
CGCNN预测脚本
"""

import os
import csv
import shutil
import datetime
import torch
import argparse
import numpy as np
from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet

# ================================================================================================
# import helper functions from change/ (prefer package import)
try:
    from cgcnn.change.filter_candidates import filter_and_save
    from cgcnn.change.parityPlot import plot_predictions
except Exception:
    # Fallback: load modules directly from the change/ script files using importlib
    import importlib.util
    change_dir = os.path.join(os.path.dirname(__file__), 'change')
    fc_path = os.path.join(change_dir, 'filter_candidates.py')
    pp_path = os.path.join(change_dir, 'parityPlot.py')

    def _load_module_from_path(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    fc_mod = _load_module_from_path('cgcnn_change_filter_candidates', fc_path)
    pp_mod = _load_module_from_path('cgcnn_change_parityPlot', pp_path)
    filter_and_save = getattr(fc_mod, 'filter_and_save')
    plot_predictions = getattr(pp_mod, 'plot_predictions')

# ================================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('modelpath',
                        help='Path to model checkpoint file, or comma-separated list of checkpoints for ensemble')
    parser.add_argument('datapath')
    parser.add_argument('--csv-output', type=str, default=None,
                        help='Optional path to write CSV output')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--disable-cuda', action='store_true')
    parser.add_argument('--ensemble-threshold', type=float, default=0.2,
                        help='Std threshold (eV) below which a prediction is considered "stable/high-confidence". Set <=0 to disable')
    args = parser.parse_args()

    # 支持单个 checkpoint 或 用逗号分隔的多 checkpoint（ensemble）
    modelpaths = [p.strip() for p in args.modelpath.split(',') if p.strip()]

    # 读取第一个 checkpoint 以构建模型结构
    first_ckpt = torch.load(modelpaths[0], map_location='cpu')
    model_args = first_ckpt.get('model_args') or first_ckpt.get('args')

    dataset = CIFData(args.datapath)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pool)

    model = CrystalGraphConvNet(
        orig_atom_fea_len=getattr(model_args, 'orig_atom_fea_len', 92),
        nbr_fea_len=getattr(model_args, 'nbr_fea_len', 41),
        atom_fea_len=getattr(model_args, 'atom_fea_len', 64),
        n_conv=getattr(model_args, 'n_conv', 3),
        h_fea_len=getattr(model_args, 'h_fea_len', 128),
        n_h=getattr(model_args, 'n_h', 1),
        classification=getattr(model_args, 'classification', False)
    )
    model.eval()
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    # 对每个模型做预测（在原始归一化空间），然后用各自 checkpoint 的 normalizer 反归一化
    all_model_preds = []  # list of lists: each inner list is predictions for all samples (denormed)
    cif_ids = None
    targets = None
    MANUAL_MEAN = 1.5972
    MANUAL_STD = 1.2327
    for mp in modelpaths:
        print(f"Loading checkpoint: {mp}")
        checkpoint = torch.load(mp, map_location='cpu')
        ckpt_normalizer = checkpoint.get('normalizer', {})
        m_mean = ckpt_normalizer.get('mean', None)
        m_std = ckpt_normalizer.get('std', None)
        if m_mean is None or m_std is None:
            print(f"  checkpoint {mp} 缺少 normalizer，使用手动或首个 checkpoint 的值。")
            if 'normalizer' in first_ckpt:
                m_mean = float(first_ckpt['normalizer'].get('mean', MANUAL_MEAN))
                m_std = float(first_ckpt['normalizer'].get('std', MANUAL_STD))
            else:
                m_mean = MANUAL_MEAN
                m_std = MANUAL_STD
        else:
            m_mean = float(m_mean)
            m_std = float(m_std)
        print(f"  normalizer mean={m_mean:.4f}, std={m_std:.4f}")

        model.load_state_dict(checkpoint['state_dict'])

        preds = []
        ids = []
        targs = []
        with torch.no_grad():
            for batch in data_loader:
                inputs, target_batch, batch_ids = batch
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
                if use_cuda:
                    atom_fea = atom_fea.cuda()
                    nbr_fea = nbr_fea.cuda()
                    nbr_fea_idx = nbr_fea_idx.cuda()
                    crystal_atom_idx = [idx.cuda() for idx in crystal_atom_idx]
                output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
                preds.extend(output.cpu().numpy().flatten().tolist())
                ids.extend(batch_ids)
                targs.extend(target_batch.cpu().numpy().flatten().tolist())

        # 反标准化（denorm）
        preds_denorm = [p * m_std + m_mean for p in preds]
        all_model_preds.append(preds_denorm)
        if cif_ids is None:
            cif_ids = ids
        if targets is None:
            targets = targs

    # 转换为 numpy array: shape (n_models, n_samples)
    preds_arr = np.array(all_model_preds, dtype=float)
    # 计算均值与标准差
    mean_preds = np.mean(preds_arr, axis=0)
    std_preds = np.std(preds_arr, axis=0)

    # 输出文件路径
    if args.csv_output:
        output_file = args.csv_output
    else:
        output_file = os.path.join(args.datapath, 'test_results_final.csv')

    conf_thresh = args.ensemble_threshold
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['id', 'target', 'prediction', 'prediction_std']
        for i in range(preds_arr.shape[0]):
            header.append(f'prediction_{i}')
        header.append('stable_prediction')
        writer.writerow(header)
        for i, mid in enumerate(cif_ids):
            tgt = float(targets[i]) if targets is not None else ''
            pred_mean = float(mean_preds[i])
            pred_std = float(std_preds[i])
            row = [mid, f"{tgt:.4f}", f"{pred_mean:.4f}", f"{pred_std:.4f}"]
            for m in range(preds_arr.shape[0]):
                row.append(f"{float(preds_arr[m, i]):.4f}")
            stable = ''
            if conf_thresh and conf_thresh > 0:
                stable = '1' if pred_std <= conf_thresh else '0'
            row.append(stable)
            writer.writerow(row)

    print(f"预测完成，结果保存至 {output_file}")

    # === 自动筛选与绘图：在仓库根目录的 log/<YYYYMMDDHHMM> 目录下保存结果 ===
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
    log_subdir = os.path.join(repo_root, 'log', timestamp)
    os.makedirs(log_subdir, exist_ok=True)

    # 复制预测结果到日志目录，避免修改原始输出
    dst_pred = os.path.join(log_subdir, os.path.basename(output_file))
    try:
        shutil.copy(output_file, dst_pred)
    except Exception:
        # 如果复制失败（例如不同文件系统），使用原始路径
        dst_pred = output_file

    # 运行筛选
    try:
        filtered_df, filtered_csv = filter_and_save(dst_pred, log_subdir)
        print(f"筛选完成，保存于: {filtered_csv}，候选数: {len(filtered_df)}")
    except Exception as e:
        print("筛选失败:", e)

    # 生成绘图并保存到日志目录
    try:
        plot_files = plot_predictions(dst_pred, log_subdir)
        print('绘图已保存：', ', '.join(plot_files))
    except Exception as e:
        print('绘图失败：', e)

if __name__ == '__main__':
    main()