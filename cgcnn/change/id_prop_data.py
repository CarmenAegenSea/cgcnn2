"""
从 CIF 目录自动生成 id_prop.csv 文件
"""

import os
from pathlib import Path

# =====================================================================================================================
# 配置参数
CIF_DIR = "data/tmc_data/cif"  # CIF文件目录
OUTPUT_FILE = "data/tmc_data/cif/id_prop.csv"  # 输出文件路径

# =====================================================================================================================
def generate_id_prop(cif_dir, output_file):
    cif_path = Path(cif_dir)
    if not cif_path.exists():
        print(f"目录不存在: {cif_dir}")
        return

    cifs = sorted(cif_path.glob("*.cif"))
    if not cifs:
        print("未找到任何 .cif 文件")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        for cif in cifs:
            material_id = cif.stem  # 去除 .cif 后缀
            f.write(f"{material_id},0\n")

    print(f"已生成 {output_file}，共 {len(cifs)} 条记录")


if __name__ == "__main__":
    generate_id_prop(CIF_DIR, OUTPUT_FILE)