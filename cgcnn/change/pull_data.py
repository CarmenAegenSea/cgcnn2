"""
从Materials Project数据库系统地拉取所有过渡金属硫族化合物(TMCs)数据
"""

import os
import json
import time
import csv
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
from itertools import product

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

# ========================================================================================================================
# 配置参数
API_KEY = "rdq9JwSE1rePyRtCKlqS6ZQgGWcYoz9U"    # MP API密钥
OUTPUT_DIR = "data/tmc_data"                    # 数据保存目录
DOWNLOAD_CIF = True                             # 是否下载CIF结构文件
SAVE_CSV = True                                 # 是否保存CSV文件
SAVE_JSON = True                                # 是否保存JSON文件
REQUEST_DELAY = 0.1                             # 请求间隔(秒)，避免触发API频率限制    

# ========================================================================================================================
# TMCs 元素定义
# 过渡金属：3-12族所有元素
TRANSITION_METALS = [
    # 第3族
    "Sc", "Y",
    # 第4族
    "Ti", "Zr", "Hf",
    # 第5族
    "V", "Nb", "Ta",
    # 第6族
    "Cr", "Mo", "W",
    # 第7族
    "Mn", "Tc", "Re",
    # 第8族
    "Fe", "Ru", "Os",
    # 第9族
    "Co", "Rh", "Ir",
    # 第10族
    "Ni", "Pd", "Pt",
    # 第11族
    "Cu", "Ag", "Au",
    # 第12族
    "Zn", "Cd", "Hg",
    # 镧系 (4f区)
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    # 锕系 (5f区)
    "Ac", "Th", "Pa", "U", "Np", "Pu"
]

# 硫族元素
CHALCOGENS = ["S", "Se", "Te"]

# 排除稀有/放射性元素
EXCLUDED_ELEMENTS: Set[str] = set()  # 如 {"Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu"}

def get_api_key() -> Optional[str]:
    if API_KEY and API_KEY != "API_KEY":
        return API_KEY
    return os.environ.get("MP_API_KEY")

# ========================================================================================================================
# 生成过渡金属-硫族元素化学体系组合，并查询Materials Project数据库获取TMCs数据
def generate_chemsys_combinations(t_metals: List[str], chalcogens: List[str]) -> List[str]:
    combinations = []
    for tm, ch in product(t_metals, chalcogens):
        # 跳过被排除的元素
        if tm in EXCLUDED_ELEMENTS or ch in EXCLUDED_ELEMENTS:
            continue
        # chemsys 格式要求元素按字母顺序排列
        chemsys = "-".join(sorted([tm, ch]))
        combinations.append(chemsys)
    return combinations

# 查询指定化学体系的TMCs数据并下载CIF文件
def query_tmc_by_chemsys(
        mpr: MPRester,
        chemsys: str,
        download_cif: bool = True,
        cif_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    materials = []

    # 需要获取的字段
    fields = [
        "material_id",
        "formula_pretty",
        "chemsys",
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
        "volume",
        "density",
        "nsites",
        "elements",
        "structure",
        "symmetry",
        "is_stable",
    ]

    try:
        docs = mpr.materials.summary.search(
            chemsys=chemsys,
            fields=fields,
        )

        for doc in docs:
            # 提取元素列表
            elem_list = []
            if hasattr(doc, 'elements') and doc.elements:
                for elem in doc.elements:
                    sym = elem.symbol if hasattr(elem, 'symbol') else str(elem)
                    elem_list.append(sym)

            # 提取晶系和空间群
            crystal_system_str = None
            spacegroup_str = None
            if hasattr(doc, 'symmetry') and doc.symmetry:
                cs = getattr(doc.symmetry, 'crystal_system', None)
                if cs is not None:
                    crystal_system_str = cs.value if hasattr(cs, 'value') else str(cs)
                sg = getattr(doc.symmetry, 'symbol', None)
                if sg is not None:
                    spacegroup_str = str(sg)

            data = {
                "material_id": str(doc.material_id),
                "formula": getattr(doc, 'formula_pretty', None),
                "chemsys": getattr(doc, 'chemsys', None),
                "band_gap": getattr(doc, 'band_gap', None),
                "formation_energy_per_atom": getattr(doc, 'formation_energy_per_atom', None),
                "energy_above_hull": getattr(doc, 'energy_above_hull', None),
                "volume": getattr(doc, 'volume', None),
                "density": getattr(doc, 'density', None),
                "nsites": getattr(doc, 'nsites', None),
                "elements": elem_list,
                "crystal_system": crystal_system_str,
                "spacegroup": spacegroup_str,
                "is_stable": getattr(doc, 'is_stable', None),
            }

            # 下载CIF文件
            if download_cif and cif_dir and hasattr(doc, 'structure') and doc.structure:
                cif_path = cif_dir / f"{doc.material_id}.cif"
                try:
                    cif_content = str(CifWriter(doc.structure))
                    with open(cif_path, 'w', encoding='utf-8') as f:
                        f.write(cif_content)
                    data["cif_path"] = str(cif_path)
                except Exception as e:
                    print(f"    保存CIF失败 {doc.material_id}: {e}")
                    data["cif_path"] = None
            else:
                data["cif_path"] = None

            materials.append(data)

        time.sleep(REQUEST_DELAY)

    except Exception as e:
        print(f"  查询 chemsys={chemsys} 时出错: {e}")

    return materials


def fetch_all_tmcs(
        mpr: MPRester,
        t_metals: List[str],
        chalcogens: List[str],
        download_cif: bool = True,
        cif_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    # 遍历所有过渡金属-硫族元素组合，拉取全部TMCs数据
    chemsys_list = generate_chemsys_combinations(t_metals, chalcogens)
    total_combinations = len(chemsys_list)

    print(f"共生成 {total_combinations} 个过渡金属-硫族元素化学体系组合")
    print("开始查询 Materials Project...")

    all_materials = []
    successful_combinations = 0
    total_materials_found = 0

    for i, chemsys in enumerate(chemsys_list, 1):
        print(f"[{i}/{total_combinations}] 查询 chemsys={chemsys}...", end=" ")

        materials = query_tmc_by_chemsys(
            mpr,
            chemsys,
            download_cif=download_cif,
            cif_dir=cif_dir,
        )

        if materials:
            print(f"找到 {len(materials)} 个材料")
            all_materials.extend(materials)
            successful_combinations += 1
            total_materials_found += len(materials)
        else:
            print("无数据")

    print(f"\n查询完成")
    print(f"  有效化学体系组合: {successful_combinations}/{total_combinations}")
    print(f"  总材料数: {total_materials_found}")

    return all_materials


def save_data(data: List[Dict[str, Any]], output_dir: Path):
    # 保存材料数据为CSV和JSON文件
    if not data:
        print("没有数据可保存")
        return

    # 移除structure字段(无法序列化)
    light_data = []
    for item in data:
        light_item = {k: v for k, v in item.items() if k != "structure"}
        light_data.append(light_item)

    if SAVE_CSV:
        csv_path = output_dir / "tmc_all_materials.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=light_data[0].keys())
            writer.writeheader()
            writer.writerows(light_data)
        print(f"CSV 已保存至: {csv_path}")

    if SAVE_JSON:
        json_path = output_dir / "tmc_all_materials.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(light_data, f, indent=2, ensure_ascii=False)
        print(f"JSON 已保存至: {json_path}")


def print_statistics(data: List[Dict[str, Any]]):
    # 打印数据统计信息
    if not data:
        return

    print("\n" + "=" * 60)
    print("TMCs 数据统计")
    print("=" * 60)
    print(f"总材料数: {len(data)}")

    # 按硫族元素统计
    chalcogen_counts = {"S": 0, "Se": 0, "Te": 0}
    for item in data:
        elems = item.get("elements", [])
        if "S" in elems:
            chalcogen_counts["S"] += 1
        if "Se" in elems:
            chalcogen_counts["Se"] += 1
        if "Te" in elems:
            chalcogen_counts["Te"] += 1

    print(f"  含S材料: {chalcogen_counts['S']}")
    print(f"  含Se材料: {chalcogen_counts['Se']}")
    print(f"  含Te材料: {chalcogen_counts['Te']}")

    # 带隙统计
    band_gaps = [d["band_gap"] for d in data if d["band_gap"] is not None]
    if band_gaps:
        print(f"  带隙范围: {min(band_gaps):.2f} - {max(band_gaps):.2f} eV")
        print(f"  平均带隙: {sum(band_gaps) / len(band_gaps):.2f} eV")

    # 形成能统计
    form_energies = [d["formation_energy_per_atom"] for d in data if d["formation_energy_per_atom"] is not None]
    if form_energies:
        print(f"  形成能范围: {min(form_energies):.3f} - {max(form_energies):.3f} eV/atom")

    # 稳定性统计
    stable_count = sum(1 for d in data if d.get("is_stable", False))
    print(f"  热力学稳定材料: {stable_count}")


def main():
    print("=" * 60)
    print("过渡金属硫族化合物 (TMCs) 数据拉取脚本")
    print("=" * 60)

    api_key = get_api_key()
    if not api_key:
        print("错误：未找到API密钥")
        print("请设置环境变量 MP_API_KEY 或修改脚本中的 API_KEY 变量")
        print("获取方式：登录 https://materialsproject.org，进入Dashboard")
        return

    output_dir = Path(OUTPUT_DIR)
    cif_dir = output_dir / "cif"
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据保存目录: {output_dir}")
    print(f"过渡金属数量: {len(TRANSITION_METALS)}")
    print(f"硫族元素: {CHALCOGENS}")
    if EXCLUDED_ELEMENTS:
        print(f"排除元素: {sorted(EXCLUDED_ELEMENTS)}")

    try:
        with MPRester(api_key) as mpr:
            all_materials = fetch_all_tmcs(
                mpr,
                TRANSITION_METALS,
                CHALCOGENS,
                download_cif=DOWNLOAD_CIF,
                cif_dir=cif_dir,
            )

            if all_materials:
                save_data(all_materials, output_dir)
                print_statistics(all_materials)

                if DOWNLOAD_CIF:
                    cif_count = len(list(cif_dir.glob("*.cif")))
                    print(f"  CIF文件数: {cif_count}")
            else:
                print("未找到任何TMCs材料，请检查网络或API密钥。")

    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()