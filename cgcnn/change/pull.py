"""
催化剂数据拉取脚本
"""

import os
import time
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

# ========================================================================================================================
# 配置参数
API_KEY = "rdq9JwSE1rePyRtCKlqS6ZQgGWcYoz9U"
OUTPUT_DIR = "data/catalysis"                   # 输出目录
MAX_MATERIALS = 20000                           # 每个通道最大材料数，最终总数可能略多于此值
REQUEST_DELAY = 0.1

MAX_ENERGY_ABOVE_HULL = 0.2                     # 排除极不稳定结构

# 排除放射性元素
EXCLUDED_ELEMENTS: Set[str] = {"Tc", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu"}

# 输出控制
DOWNLOAD_CIF = True
SAVE_CSV = True
SAVE_JSON = True

# 带隙下限过滤：低于此值的材料视为金属/半金属，不保存
BAND_GAP_MIN = 1e-6

# ========================================================================================================================
# 获取API密钥
def get_api_key() -> Optional[str]:
    if API_KEY and API_KEY != "api_key":
        return API_KEY
    return os.environ.get("MP_API_KEY")

def element_blacklist_filter(elements: List[str]) -> bool:
    if not elements:
        return False
    return any(elem in EXCLUDED_ELEMENTS for elem in elements)

# =========================================================================================================================
# 查询部分
# 通过元素组合查询（限定元素必须都存在）
def fetch_materials_by_elements(mpr: MPRester, elements_list: List[List[str]], description: str) -> List[Dict]:
    print(f"\n查询：{description}")
    materials = []
    for elem_set in elements_list:
        print(f"  元素组合: {elem_set}")
        try:
            docs = mpr.materials.summary.search(
                elements=elem_set,
                energy_above_hull=(None, MAX_ENERGY_ABOVE_HULL),
                fields=["material_id", "formula_pretty", "band_gap", "formation_energy_per_atom",
                        "energy_above_hull", "volume", "density", "nsites", "elements",
                        "cbm", "vbm", "is_stable", "symmetry", "structure"]
            )
            for doc in docs:
                if MAX_MATERIALS is not None and len(materials) >= MAX_MATERIALS // 2:   # 限制每个通道数量，避免失衡
                    break
                materials.append(process_doc(doc))
                time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"    查询失败: {e}")
    print(f"  找到 {len(materials)} 个材料")
    return materials

# 通过化学体系查询（精确匹配化学体系）
def fetch_materials_by_chemsys(mpr: MPRester, chemsys_list: List[str], description: str) -> List[Dict]:
    print(f"\n查询：{description}")
    materials = []
    for chemsys in chemsys_list:
        print(f"  化学体系: {chemsys}")
        try:
            docs = mpr.materials.summary.search(
                chemsys=chemsys,
                energy_above_hull=(None, MAX_ENERGY_ABOVE_HULL),
                fields=["material_id", "formula_pretty", "band_gap", "formation_energy_per_atom",
                        "energy_above_hull", "volume", "density", "nsites", "elements",
                        "cbm", "vbm", "is_stable", "symmetry", "structure"]
            )
            for doc in docs:
                if len(materials) >= MAX_MATERIALS // 3:
                    break
                materials.append(process_doc(doc))
                time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"    查询失败: {e}")
    print(f"  找到 {len(materials)} 个材料")
    return materials

# ========================================================================================================================
# 处理 API 返回的文档，提取关键信息并转换为统一格式
def process_doc(doc) -> Dict[str, Any]:
    elem_list = [e.symbol if hasattr(e, 'symbol') else str(e) for e in doc.elements]
    crystal_system_str = None
    spacegroup_str = None
    if hasattr(doc, 'symmetry') and doc.symmetry:
        cs = getattr(doc.symmetry, 'crystal_system', None)
        if cs is not None:
            crystal_system_str = cs.value if hasattr(cs, 'value') else str(cs)
        sg = getattr(doc.symmetry, 'symbol', None)
        if sg is not None:
            spacegroup_str = str(sg)

    return {
        "material_id": str(doc.material_id),
        "formula": getattr(doc, 'formula_pretty', None),
        "band_gap": getattr(doc, 'band_gap', None),
        "formation_energy_per_atom": getattr(doc, 'formation_energy_per_atom', None),
        "energy_above_hull": getattr(doc, 'energy_above_hull', None),
        "volume": getattr(doc, 'volume', None),
        "density": getattr(doc, 'density', None),
        "nsites": getattr(doc, 'nsites', None),
        "elements": elem_list,
        "cbm": getattr(doc, 'cbm', None),
        "vbm": getattr(doc, 'vbm', None),
        "crystal_system": crystal_system_str,
        "spacegroup": spacegroup_str,
        "is_stable": getattr(doc, 'is_stable', None),
        "structure": doc.structure if hasattr(doc, 'structure') else None,
    }

def save_candidates(candidates: List[Dict[str, Any]], output_dir: Path):
    if not candidates:
        print("没有候选材料，跳过保存。")
        return

    total_before_filter = len(candidates)
    candidates = [
        c for c in candidates
        if c.get("band_gap") is not None and c["band_gap"] > BAND_GAP_MIN
    ]
    removed = total_before_filter - len(candidates)
    if removed > 0:
        print(f"已过滤掉 {removed} 个 band_gap <= {BAND_GAP_MIN} (含 0 或金属性) 的材料")

    if not candidates:
        print("过滤后没有剩余材料，跳过保存。")
        return

    cif_dir = output_dir / "cif"
    cif_dir.mkdir(parents=True, exist_ok=True)

    light_data = []
    for item in candidates:
        light_item = {k: v for k, v in item.items() if k != "structure"}
        light_data.append(light_item)

        if DOWNLOAD_CIF and item.get("structure"):
            cif_path = cif_dir / f"{item['material_id']}.cif"
            try:
                cif_string = str(CifWriter(item["structure"]))
                with open(cif_path, 'w', encoding='utf-8') as f:
                    f.write(cif_string)
                light_item["cif_path"] = str(cif_path)
            except Exception as e:
                print(f"  保存 CIF 失败 {item['material_id']}: {e}")
                light_item["cif_path"] = None
        else:
            light_item["cif_path"] = None

    if SAVE_CSV:
        csv_path = output_dir / "catalysis.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=light_data[0].keys())
            writer.writeheader()
            writer.writerows(light_data)
        print(f"CSV 已保存至: {csv_path}")

    if SAVE_JSON:
        json_path = output_dir / "catalysis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(light_data, f, indent=2, ensure_ascii=False)
        print(f"JSON 已保存至: {json_path}")

    print(f"CIF 文件总数: {len(list(cif_dir.glob('*.cif')))}")

def main():
    print("=" * 60)
    print("Materials Project 已验证光催化剂数据拉取")
    print("=" * 60)

    api_key = get_api_key()
    if not api_key:
        print("错误：未找到 API 密钥！")
        return

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"数据保存目录: {output_dir}")

    try:
        with MPRester(api_key) as mpr:
            all_materials = []
            material_ids = set()

            # 通道1：经典氧化物光催化剂 (TiO2, ZnO, WO3, Fe2O3, Cu2O, BiVO4, SnO2, NiO, Co3O4, MnO2)
            oxide_elements = [
                ["Ti", "O"], ["Zn", "O"], ["W", "O"], ["Fe", "O"],
                ["Cu", "O"], ["Bi", "V", "O"], ["Sn", "O"],
                ["Ni", "O"], ["Co", "O"], ["Mn", "O"]
            ]
            oxides = fetch_materials_by_elements(mpr, oxide_elements, "经典金属氧化物光催化剂")
            for mat in oxides:
                if mat['material_id'] not in material_ids:
                    material_ids.add(mat['material_id'])
                    all_materials.append(mat)

            # 通道2：经典硫化物光催化剂 (CdS, ZnS, MoS2, WS2, Cu2S, Bi2S3, Sb2S3, SnS2, ZnIn2S4, CZTS)
            sulfide_elements = [
                ["Cd", "S"], ["Zn", "S"], ["Mo", "S"], ["W", "S"],
                ["Cu", "S"], ["Bi", "S"], ["Sb", "S"], ["Sn", "S"],
                ["Zn", "In", "S"], ["Cu", "Zn", "Sn", "S"]
            ]
            sulfides = fetch_materials_by_elements(mpr, sulfide_elements, "经典金属硫化物光催化剂")
            for mat in sulfides:
                if mat['material_id'] not in material_ids:
                    material_ids.add(mat['material_id'])
                    all_materials.append(mat)

            # 通道3：氮化物/氮氧化物 (Ta3N5, NbN, TiN, WN, GaN, InN, Zn3N2, Ge3N4, LaTiO2N, BaTaO2N)
            nitride_chemsys = [
                "N-Ta", "N-Nb", "N-Ti", "N-W", "N-Mo",
                "N-Ga", "N-In", "N-Zn", "N-Ge",
                "La-Ti-O-N", "Ba-Ta-O-N"
            ]
            nitrides = fetch_materials_by_chemsys(mpr, nitride_chemsys, "氮化物/氮氧化物光催化剂")
            for mat in nitrides:
                if mat['material_id'] not in material_ids:
                    material_ids.add(mat['material_id'])
                    all_materials.append(mat)

            # 通道4：层状双氢氧化物 LDH (Mg-Al, Zn-Al, Ni-Fe, Ni-Al, Co-Al, Li-Al, Zn-Cr, Cu-Zn-Al)
            ldh_chemsys = [
                "Mg-Al-O-H", "Zn-Al-O-H", "Ni-Fe-O-H", "Ni-Al-O-H",
                "Co-Al-O-H", "Li-Al-O-H", "Zn-Cr-O-H", "Cu-Zn-Al-O-H"
            ]
            ldhs = fetch_materials_by_chemsys(mpr, ldh_chemsys, "层状双氢氧化物 (LDH) 光催化剂")
            for mat in ldhs:
                if mat['material_id'] not in material_ids:
                    material_ids.add(mat['material_id'])
                    all_materials.append(mat)

            # 通道5：MXene 前驱体 (MAX 相) (Ti2AlC, Ti2AlN, V2AlC, Cr2AlC, Mo2AlC, Nb2AlC)
            mxene_elements = [
                ["Ti", "Al", "C"], ["Ti", "Al", "N"], ["V", "Al", "C"],
                ["Cr", "Al", "C"], ["Mo", "Al", "C"], ["Nb", "Al", "C"]
            ]
            mxenes = fetch_materials_by_elements(mpr, mxene_elements, "MXene 前驱体 (MAX相)")
            for mat in mxenes:
                if mat['material_id'] not in material_ids:
                    material_ids.add(mat['material_id'])
                    all_materials.append(mat)

            # 钙钛矿型光催化剂 (SrTiO3, BaTiO3, CaTiO3, LaFeO3, BiFeO3, NaTaO3, KTaO3, KNbO3 等)
            perovskite_chemsys = [
                "Sr-Ti-O", "Ba-Ti-O", "Ca-Ti-O", "La-Fe-O",
                "Bi-Fe-O", "Na-Ta-O", "K-Ta-O", "K-Nb-O",
                "Y-Fe-O", "La-Co-O", "La-Mn-O", "La-Ni-O"
            ]
            perovskites = fetch_materials_by_chemsys(mpr, perovskite_chemsys, "钙钛矿型光催化剂")
            for mat in perovskites:
                if mat['material_id'] not in material_ids:
                    material_ids.add(mat['material_id'])
                    all_materials.append(mat)

            # g-C3N4 及其衍生物 (C-N, C-N-H, C-N-O)
            gcn_chemsys = ["C-N", "C-N-H", "C-N-O"]
            gcns = fetch_materials_by_chemsys(mpr, gcn_chemsys, "g-C3N4 及其衍生物")
            for mat in gcns:
                if mat['material_id'] not in material_ids:
                    material_ids.add(mat['material_id'])
                    all_materials.append(mat)

            # 截断至 MAX_MATERIALS
            if MAX_MATERIALS is not None:
                all_materials = all_materials[:MAX_MATERIALS]

            print("\n" + "=" * 60)
            print(f"最终收集材料总数: {len(all_materials)}")
            print("=" * 60)

            if all_materials:
                save_candidates(all_materials, output_dir)
                band_gaps = [c["band_gap"] for c in all_materials if c["band_gap"] is not None and c["band_gap"] > BAND_GAP_MIN]
                if band_gaps:
                    print(f"带隙范围: {min(band_gaps):.2f} - {max(band_gaps):.2f} eV")
            else:
                print("未找到任何候选材料。")

    except Exception as e:
        print(f"执行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()