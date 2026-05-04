"""
交叉验证数据划分脚本
将 CIF 文件随机均匀分配到 5 个文件夹内
"""

import os
import random
import shutil
import pandas as pd

# ================================================================================================================================
# 配置参数
# 源文件夹（包含 .cif 文件的目录）
YUAN_DIR = r"data/catalysis/cif"                       
# 总表id_prop.csv的路径                                              
MASTER_CSV = r"data/catalysis/cif/id_prop.csv"        
# 目标目录
OUTPUT_ROOT = r"data/catalysis/cif"       
# catalysis.json源文件路径          
ATOM_INIT_SRC = r"data/catalysis/catalysis.json"     
# 随机种子         
RANDOM_SEED = 42                                                                                

# ================================================================================================================================
def main():
    # 检查路径
    for path, name in [(YUAN_DIR, "YUAN_DIR"), (MASTER_CSV, "MASTER_CSV"), (OUTPUT_ROOT, "OUTPUT_ROOT")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"路径不存在：{name} = {path}")
    if not os.path.exists(ATOM_INIT_SRC):
        print(f"警告：atom_init.json 文件不存在于 {ATOM_INIT_SRC}，将跳过复制。")

    # 读取总表
    master_df = pd.read_csv(MASTER_CSV, header=None, names=["id", "target"])
    master_df["id"] = master_df["id"].astype(str)
    data_map = dict(zip(master_df["id"], master_df["target"]))
    print(f"已从总表读取 {len(data_map)} 条数据。")

    # 收集 CIF 文件（扩展名匹配大小写不敏感）
    cif_files = [f[:-4] for f in os.listdir(YUAN_DIR) if f.lower().endswith(".cif")]
    print(f"找到 {len(cif_files)} 个 .cif 文件。")

    # 过滤出有标签的材料
    valid_ids = [cid for cid in cif_files if cid in data_map]
    print(f"其中有标签的有效材料数：{len(valid_ids)}")
    if not valid_ids:
        raise RuntimeError("没有有效材料，请检查 CIF 文件名与 id_prop.csv 中的 ID 是否匹配。")

    # 随机打乱
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    random.shuffle(valid_ids)

    # 均匀分成 5 份
    n = len(valid_ids)
    splits = []
    for i in range(5):
        start = i * n // 5
        end = (i + 1) * n // 5
        splits.append(valid_ids[start:end])

    # 创建 5 个文件夹并写入数据
    for i in range(1, 6):
        fold_dir = os.path.join(OUTPUT_ROOT, str(i))
        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)
        os.makedirs(fold_dir)

        ids = splits[i - 1]
        print(f"Fold {i}: {len(ids)} 个材料")

        fold_data = []
        for cid in ids:
            src_cif = os.path.join(YUAN_DIR, cid + ".cif")
            dst_cif = os.path.join(fold_dir, cid + ".cif")
            shutil.copy2(src_cif, dst_cif)
            fold_data.append([cid, data_map[cid]])

        # 写入 id_prop.csv
        fold_csv = os.path.join(fold_dir, "id_prop.csv")
        pd.DataFrame(fold_data).to_csv(fold_csv, index=False, header=False)

        # 复制 atom_init.json
        if os.path.exists(ATOM_INIT_SRC):
            shutil.copy2(ATOM_INIT_SRC, os.path.join(fold_dir, "atom_init.json"))
        else:
            print(f"  警告：未复制 atom_init.json 到 Fold {i}")

    print("\n随机分配完成，生成的文件夹：")
    for i in range(1, 6):
        print(f"  {os.path.join(OUTPUT_ROOT, str(i))}")

if __name__ == "__main__":
    main()