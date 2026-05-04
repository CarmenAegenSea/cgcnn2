"""
交叉验证脚本
"""

import os
import sys
import subprocess
import shutil
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# =======================================================================================================
# 配置参数
SPLIT_BASE = r"C:\\Users\\22616\\PycharmProjects\\cgcnn2\\data\\catalysis_split"    # 预分割根目录
WORK_BASE = "data/kfold_temp"                                                       # 临时工作根目录
N_FOLDS = 5                                                                         # 折数
CGCNN_MAIN = os.path.join(os.path.dirname(__file__), "main.py")
CGCNN_PREDICT = os.path.join(os.path.dirname(__file__), "predict_data.py")
USE_CUDA = True

# 训练参数
EXTRA_ARGS = [
    "--epochs", "500",
    "--batch-size", "256",
    "--lr", "0.001",
    "--optim", "Adam",
]

# 内部验证比例（从训练集中划分，用于早停）
INTERNAL_VAL_RATIO = 0.1

# 预测输出文件名（与predict_data.py输出一致）
PREDICT_OUTPUT_FILE = "test_results_final.csv"

# ======================================================================================================
CGCNN_MAIN_ABS = os.path.abspath(CGCNN_MAIN)
CGCNN_PREDICT_ABS = os.path.abspath(CGCNN_PREDICT)

# ======================================================================================================
def prepare_train_dir(fold_idx, train_folders, work_base):
    """将多个训练文件夹的数据合并到临时训练目录"""
    train_dir = os.path.join(work_base, str(fold_idx), "train")
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)

    # 合并训练集的 id_prop.csv
    train_dfs = []
    for folder in train_folders:
        csv_path = os.path.join(folder, "id_prop.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, header=None, names=["id", "target"])
            train_dfs.append(df)
        else:
            print(f"  警告：训练文件夹 {folder} 中未找到 id_prop.csv")
    if not train_dfs:
        raise RuntimeError("没有可用的训练数据")
    train_combined = pd.concat(train_dfs, ignore_index=True)
    train_combined.to_csv(os.path.join(train_dir, "id_prop.csv"), index=False, header=False)
    print(f"  训练集样本数: {len(train_combined)}")

    # 复制所有训练文件夹中的.cif文件
    copied = 0
    for folder in train_folders:
        for fname in os.listdir(folder):
            if fname.lower().endswith(".cif"):
                src = os.path.join(folder, fname)
                dst = os.path.join(train_dir, fname)
                if not os.path.exists(dst):   # 避免重复复制（若不同文件夹有同名文件，理论不应出现）
                    shutil.copy2(src, dst)
                    copied += 1
    print(f"  复制了 {copied} 个 .cif 文件到训练目录")

    # 复制 atom_init.json（从第一个文件夹复制）
    atom_src = os.path.join(train_folders[0], "atom_init.json")
    if os.path.exists(atom_src):
        shutil.copy(atom_src, train_dir)
    else:
        # 尝试其他文件夹
        for folder in train_folders:
            alt = os.path.join(folder, "atom_init.json")
            if os.path.exists(alt):
                shutil.copy(alt, train_dir)
                break
        else:
            raise FileNotFoundError("未找到 atom_init.json")
    return train_dir

def run_cgcnn(train_dir, train_size):
    """训练模型，内部验证集从训练集中按比例划分"""
    val_size = max(1, int(train_size * INTERNAL_VAL_RATIO))
    train_size_final = train_size - val_size
    cmd = [sys.executable, CGCNN_MAIN_ABS,
           "--train-size", str(train_size_final),
           "--val-size", str(val_size),
           "--test-size", "0",               # 不使用独立测试集
           os.path.abspath(train_dir)]
    if not USE_CUDA:
        cmd.append("--disable-cuda")
    cmd.extend(EXTRA_ARGS)
    print("Training command:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=train_dir)

def run_predict_for_val(val_folder, result_list, fold_idx, work_base):
    """对验证文件夹进行预测，收集结果"""
    # 先在验证文件夹上运行预测脚本
    model_abspath = os.path.abspath(os.path.join(work_base, str(fold_idx), "train", "model_best.pth.tar"))
    cmd = [sys.executable, CGCNN_PREDICT_ABS,
           model_abspath,
           os.path.abspath(val_folder)]
    if not USE_CUDA:
        cmd.append("--disable-cuda")
    print("Predict command:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=val_folder)

    # 读取预测结果
    pred_file = os.path.join(val_folder, PREDICT_OUTPUT_FILE)
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"预测输出未找到: {pred_file}")
    
    # 修改这里：不指定header，先读进来
    pred_df = pd.read_csv(pred_file)
    
    print(f"  预测文件实际列名: {list(pred_df.columns)}")
    
    # 只取前三列，并强制重命名
    pred_df = pred_df.iloc[:, :3] 
    pred_df.columns = ["id", "target", "prediction"]
    
    pred_df["id"] = pred_df["id"].astype(str)
    result_list.append(pred_df)
    print(f"  验证集预测样本数: {len(pred_df)}")

def patch_model_checkpoint(fold_work_dir):
    """修补模型参数以兼容 predict.py"""
    model_path = os.path.join(fold_work_dir, "train", "model_best.pth.tar")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    try:
        import torch
    except ImportError:
        raise RuntimeError("修补检查点需要 PyTorch，请确保在对应环境中运行")
    checkpoint = torch.load(model_path, map_location='cpu')
    args = checkpoint.get('model_args') or checkpoint.get('args') or {}
    patched = False
    if isinstance(args, dict):
        if 'atom_fea_len' in args and 'orig_atom_fea_len' not in args:
            args['orig_atom_fea_len'] = args['atom_fea_len']
            checkpoint['args'] = args
            patched = True
    else:
        if hasattr(args, 'atom_fea_len') and not hasattr(args, 'orig_atom_fea_len'):
            setattr(args, 'orig_atom_fea_len', getattr(args, 'atom_fea_len'))
            checkpoint['args'] = args
            patched = True
    if patched:
        torch.save(checkpoint, model_path)
        val = args['atom_fea_len'] if isinstance(args, dict) else getattr(args, 'atom_fea_len', None)
        print(f"模型已修补: 添加 orig_atom_fea_len = {val}")
    else:
        print("模型检查点无需修补")

def main():
    # 验证分割目录存在
    if not os.path.exists(SPLIT_BASE):
        raise FileNotFoundError(f"分割根目录不存在: {SPLIT_BASE}")
    fold_names = [str(i) for i in range(1, N_FOLDS+1)]
    for name in fold_names:
        if not os.path.isdir(os.path.join(SPLIT_BASE, name)):
            raise FileNotFoundError(f"缺少文件夹: {os.path.join(SPLIT_BASE, name)}")

    # 清理并创建工作根目录
    if os.path.exists(WORK_BASE):
        shutil.rmtree(WORK_BASE)
    os.makedirs(WORK_BASE)

    all_results = []

    for fold_idx in range(1, N_FOLDS+1):
        print(f"\n{'='*50}")
        print(f"开始第 {fold_idx} 折")
        val_folder = os.path.join(SPLIT_BASE, str(fold_idx))
        train_folders = [os.path.join(SPLIT_BASE, str(j)) for j in range(1, N_FOLDS+1) if j != fold_idx]

        # 准备训练目录（合并训练数据）
        train_dir = prepare_train_dir(fold_idx, train_folders, WORK_BASE)
        # 读取训练集样本数
        train_df = pd.read_csv(os.path.join(train_dir, "id_prop.csv"), header=None, names=["id", "target"])
        train_size = len(train_df)

        # 训练
        run_cgcnn(train_dir, train_size)

        # 修补检查点
        patch_model_checkpoint(os.path.join(WORK_BASE, str(fold_idx)))

        # 在验证文件夹上预测
        run_predict_for_val(val_folder, all_results, fold_idx, WORK_BASE)

        print(f"第 {fold_idx} 折训练及预测完成\n")

    # 汇总结果
    final_df = pd.concat(all_results, ignore_index=True)
    final_df = final_df.sort_values("id")
    out_path = os.path.join(WORK_BASE, "test_results_cv.csv")
    final_df.to_csv(out_path, index=False)

    mae = mean_absolute_error(final_df["target"], final_df["prediction"])
    print(f"所有折训练完毕！汇总预测结果已保存至: {out_path}")
    print(f"5 折交叉验证 MAE: {mae:.4f} eV")

if __name__ == "__main__":
    main()