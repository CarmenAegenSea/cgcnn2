"""
从 CIF 目录自动生成 id_prop.csv 文件
从已有的材料属性 CSV 中匹配真实带隙值
未匹配到的材料才使用占位值 0，并给出警告
"""

import os
import pandas as pd
from pathlib import Path

# =====================================================================================================================
# 配置参数
CIF_DIR = r"C:\Users\22616\PycharmProjects\cgcnn2\data\catalysis\cif"                          # CIF 文件目录
OUTPUT_FILE = "data/catalysis/cif/id_prop.csv"           # 输出文件路径
SOURCE_CSV = "data/catalysis/catalysis.csv"               # 属性表路径

# =====================================================================================================================

def generate_id_prop(cif_dir, output_file, source_csv=None):
    cif_path = Path(cif_dir)
    if not cif_path.exists():
        print(f"目录不存在: {cif_dir}")
        return

    cifs = sorted(cif_path.glob("*.cif"))
    if not cifs:
        print("未找到任何 .cif 文件")
        return

    # 尝试从属性表中加载真实带隙值
    bandgap_map = {}
    if source_csv and Path(source_csv).exists():
        df = pd.read_csv(source_csv)
        # 兼容可能的列名
        id_col = "material_id" if "material_id" in df.columns else df.columns[0]
        bg_col = "band_gap" if "band_gap" in df.columns else df.columns[1]
        bandgap_map = dict(zip(df[id_col].astype(str), df[bg_col]))
        print(f"从 {source_csv} 加载了 {len(bandgap_map)} 条带隙记录。")
    else:
        print(f"警告：未找到属性表 {source_csv}，所有目标值将用 0 占位。")

    missing = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for cif in cifs:
            material_id = cif.stem  # 去除 .cif 后缀
            if material_id in bandgap_map:
                bandgap = bandgap_map[material_id]
            else:
                bandgap = 0
                missing += 1
                print(f"  警告：{material_id} 在属性表中未找到，使用占位值 0")
            f.write(f"{material_id},{bandgap}\n")

    print(f"已生成 {output_file}，共 {len(cifs)} 条记录。")
    if missing > 0:
        print(f"其中 {missing} 条未找到真实带隙，已用 0 占位。")

if __name__ == "__main__":
    generate_id_prop(CIF_DIR, OUTPUT_FILE, SOURCE_CSV)