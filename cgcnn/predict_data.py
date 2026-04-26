"""
CGCNN预测脚本
"""

import os
import csv
import shutil
import datetime
import torch
import argparse
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
    parser.add_argument('modelpath')
    parser.add_argument('datapath')
    parser.add_argument('--csv-output', type=str, default=None,
                        help='Optional path to write CSV output')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--disable-cuda', action='store_true')
    args = parser.parse_args()

    checkpoint = torch.load(args.modelpath, map_location='cpu')
    model_args = checkpoint.get('model_args') or checkpoint.get('args')

    # 从 checkpoint 读取标准化参数
    normalizer = checkpoint.get('normalizer', {})
    mean = normalizer.get('mean')
    std = normalizer.get('std')

    if mean is None or std is None:
        print("Checkpoint 中未找到 normalizer，使用手动指定值。")
        MANUAL_MEAN = 1.5972  #替换为获取的mean
        MANUAL_STD = 1.2327   #替换为获取的std
        mean = MANUAL_MEAN
        std = MANUAL_STD
    else:
        # 确保为 Python float
        mean = float(mean)
        std = float(std)
        print(f"从 checkpoint 读取: mean = {mean:.4f}, std = {std:.4f}")

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
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    predictions = []
    cif_ids = []
    targets = []
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
            predictions.extend(output.cpu().numpy().flatten().tolist())
            cif_ids.extend(batch_ids)
            targets.extend(target_batch.cpu().numpy().flatten().tolist())

    # 逆标准化预测值
    predictions = [p * std + mean for p in predictions]

    # 输出文件路径
    if args.csv_output:
        output_file = args.csv_output
    else:
        output_file = os.path.join(args.datapath, 'test_results_final.csv')

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'target', 'prediction'])
        for mid, tgt, pred in zip(cif_ids, targets, predictions):
            writer.writerow([mid, f"{tgt:.4f}", f"{pred:.4f}"])

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