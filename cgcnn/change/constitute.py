"""
模拟生成复合光催化催化剂并使用CGCNN模型预测带隙。

该脚本从现有光催化剂数据集中随机选择材料对进行组合，
生成复合光催化材料，并使用训练好的CGCNN模型预测其带隙。

 python "C:\Users\22616\PycharmProjects\cgcnn2\cgcnn\change\constitute.py" --n 50 --seed 42
"""

import os
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import csv
import json
import datetime
import random
import shutil
import warnings
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from pymatgen.core import Structure, Lattice, Composition
from pymatgen.io.cif import CifWriter

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet

try:
    from cgcnn.change.filter_composites import filter_and_save
except Exception:
    try:
        import importlib.util
        fc_path = os.path.join(os.path.dirname(__file__), 'filter_composites.py')
        spec = importlib.util.spec_from_file_location('filter_composites', fc_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        filter_and_save = getattr(mod, 'filter_and_save')
    except Exception:
        filter_and_save = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DEFAULT_CATALYSIS_CSV = os.path.join(REPO_ROOT, 'data', 'catalysis', 'catalysis.csv')
DEFAULT_CATALYSIS_CIF = os.path.join(REPO_ROOT, 'data', 'catalysis', 'cif')
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, 'models', 'ensemble')

EXCLUDED_ELEMENTS = {"Tc", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu"}


def load_catalysis_data(csv_path):
    """加载催化剂数据集"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"催化剂CSV文件未找到: {csv_path}")

    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    cif_dir = os.path.join(csv_dir, 'cif')
    materials = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                elements = eval(row.get('elements', '[]'))
                if any(elem in EXCLUDED_ELEMENTS for elem in elements):
                    continue
                material_id = row.get('material_id', '')
                cif_path = os.path.join(cif_dir, f"{material_id}.cif")
                materials.append({
                    'material_id': material_id,
                    'formula': row.get('formula', ''),
                    'band_gap': float(row.get('band_gap', 0)) if row.get('band_gap') else 0.0,
                    'formation_energy_per_atom': float(row.get('formation_energy_per_atom', 0)),
                    'energy_above_hull': float(row.get('energy_above_hull', 0)),
                    'is_stable': row.get('is_stable', 'False') == 'True',
                    'elements': elements,
                    'cif_path': cif_path,
                    'cif_dir': cif_dir,
                    'crystal_system': row.get('crystal_system', ''),
                })
            except (ValueError, SyntaxError, TypeError):
                continue
    return materials


def generate_composite_structure(mat1, mat2):
    """通过混合两个材料的元素生成复合结构"""
    try:
        cif_path1 = mat1.get('cif_path', '')
        cif_path2 = mat2.get('cif_path', '')

        if not cif_path1 or not os.path.exists(cif_path1):
            return None
        if not cif_path2 or not os.path.exists(cif_path2):
            return None

        struct1 = Structure.from_file(cif_path1)
        struct2 = Structure.from_file(cif_path2)

        species = []
        frac_coords = []
        for site in struct1:
            species.append(site.specie)
            frac_coords.append(site.frac_coords)
        for site in struct2:
            species.append(site.specie)
            frac_coords.append(site.frac_coords)

        mixed_structure = Structure(
            struct1.lattice,
            species,
            frac_coords,
            coords_are_cartesian=False,
            validate_proximity=False
        )

        return mixed_structure
    except Exception:
        return None


def create_composite_cif(structure, composite_id, output_dir):
    """保存复合材料的CIF文件"""
    try:
        cif_path = os.path.join(output_dir, f"{composite_id}.cif")
        writer = CifWriter(structure)
        with open(cif_path, 'w', encoding='utf-8') as f:
            f.write(writer.__str__())
        return cif_path
    except Exception:
        return None


def load_cgcnn_models(model_dir):
    """加载所有ensemble模型"""
    modelpaths = []
    if os.path.exists(model_dir):
        for subdir in os.listdir(model_dir):
            subpath = os.path.join(model_dir, subdir)
            if os.path.isdir(subpath):
                best_model = os.path.join(subpath, 'model_best.pth.tar')
                if os.path.exists(best_model):
                    modelpaths.append(best_model)

    if not modelpaths:
        checkpoint = os.path.join(model_dir, 'model_best.pth.tar')
        if os.path.exists(checkpoint):
            modelpaths.append(checkpoint)

    return modelpaths


def predict_bandgaps(data_dir, modelpaths, batch_size=64, disable_cuda=False):
    """使用CGCNN模型预测带隙"""
    dataset = CIFData(data_dir)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pool
    )

    first_ckpt = torch.load(modelpaths[0], map_location='cpu')
    model_args = first_ckpt.get('model_args') or first_ckpt.get('args')

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
    use_cuda = not disable_cuda and torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    MANUAL_MEAN = 1.5972
    MANUAL_STD = 1.2327

    all_model_preds = []
    cif_ids = None

    for mp in modelpaths:
        checkpoint = torch.load(mp, map_location='cpu')
        ckpt_normalizer = checkpoint.get('normalizer', {})
        m_mean = ckpt_normalizer.get('mean', None)
        m_std = ckpt_normalizer.get('std', None)

        if m_mean is None or m_std is None:
            if 'normalizer' in first_ckpt:
                m_mean = float(first_ckpt['normalizer'].get('mean', MANUAL_MEAN))
                m_std = float(first_ckpt['normalizer'].get('std', MANUAL_STD))
            else:
                m_mean = MANUAL_MEAN
                m_std = MANUAL_STD
        else:
            m_mean = float(m_mean)
            m_std = float(m_std)

        model.load_state_dict(checkpoint['state_dict'])

        preds = []
        ids = []
        with torch.no_grad():
            for batch in data_loader:
                inputs, _, batch_ids = batch
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
                if use_cuda:
                    atom_fea = atom_fea.cuda()
                    nbr_fea = nbr_fea.cuda()
                    nbr_fea_idx = nbr_fea_idx.cuda()
                    crystal_atom_idx = [idx.cuda() for idx in crystal_atom_idx]
                output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
                preds.extend(output.cpu().numpy().flatten().tolist())
                ids.extend(batch_ids)

        preds_denorm = [p * m_std + m_mean for p in preds]
        all_model_preds.append(preds_denorm)
        if cif_ids is None:
            cif_ids = ids

    preds_arr = np.array(all_model_preds, dtype=float)
    mean_preds = np.mean(preds_arr, axis=0)
    std_preds = np.std(preds_arr, axis=0)

    return list(zip(cif_ids, mean_preds, std_preds))


def generate_and_predict(n_composites=100, seed=42, model_dir=None, catalysis_csv=None, catalysis_cif=None):
    """生成复合催化剂并预测带隙"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    if catalysis_csv is None:
        catalysis_csv = DEFAULT_CATALYSIS_CSV
    if catalysis_cif is None:
        catalysis_cif = DEFAULT_CATALYSIS_CIF

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
    output_dir = os.path.join(REPO_ROOT, 'doubleLog', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print(f"输出目录: {output_dir}")
    print(f"生成 {n_composites} 个复合光催化材料...")

    materials = load_catalysis_data(catalysis_csv)
    print(f"加载了 {len(materials)} 个光催化剂")

    if len(materials) < 2:
        print("数据集中的材料不足，无法生成复合物")
        return output_dir

    composites_dir = os.path.join(output_dir, 'composites_cif')
    os.makedirs(composites_dir, exist_ok=True)

    composites_data = []
    composite_id_set = set()

    for i in range(n_composites):
        mat1_idx = random.randint(0, len(materials) - 1)
        mat2_idx = random.randint(0, len(materials) - 1)
        while mat2_idx == mat1_idx:
            mat2_idx = random.randint(0, len(materials) - 1)

        mat1 = materials[mat1_idx]
        mat2 = materials[mat2_idx]

        composite_id = f"comp_{mat1['material_id']}_{mat2['material_id']}"
        if composite_id in composite_id_set:
            composite_id = f"comp_{i}_{mat1['material_id']}_{mat2['material_id']}"
        composite_id_set.add(composite_id)

        structure = generate_composite_structure(mat1, mat2)
        if structure is None:
            continue

        cif_path = create_composite_cif(structure, composite_id, composites_dir)
        if cif_path is None:
            continue

        combined_elements = list(set(mat1['elements']) | set(mat2['elements']))
        composites_data.append({
            'composite_id': composite_id,
            'material1_id': mat1['material_id'],
            'material2_id': mat2['material_id'],
            'material1_formula': mat1['formula'],
            'material2_formula': mat2['formula'],
            'material1_bandgap': mat1['band_gap'],
            'material2_bandgap': mat2['band_gap'],
            'elements': combined_elements,
            'cif_path': cif_path,
        })

        if (i + 1) % 20 == 0:
            print(f"  已生成 {i + 1}/{n_composites} 个复合物")

    print(f"成功生成 {len(composites_data)} 个复合物")

    if not composites_data:
        print("没有成功生成任何复合物")
        return output_dir

    id_prop_path = os.path.join(composites_dir, 'id_prop.csv')
    with open(id_prop_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for comp in composites_data:
            writer.writerow([comp['composite_id'], 2.0])

    atom_init_src = os.path.join(catalysis_cif, 'atom_init.json')
    atom_init_dst = os.path.join(composites_dir, 'atom_init.json')
    if os.path.exists(atom_init_src):
        shutil.copy(atom_init_src, atom_init_dst)

    print("开始预测带隙...")
    modelpaths = load_cgcnn_models(model_dir)
    if not modelpaths:
        print("未找到模型文件，使用默认参数初始化模型")
        modelpaths = []

    if modelpaths:
        predictions = predict_bandgaps(composites_dir, modelpaths)
        pred_dict = {pid: (mean, std) for pid, mean, std in predictions}
    else:
        pred_dict = {}

    results = []
    for comp in composites_data:
        comp_id = comp['composite_id']
        if comp_id in pred_dict:
            pred_mean, pred_std = pred_dict[comp_id]
        else:
            avg_gap = (comp['material1_bandgap'] + comp['material2_bandgap']) / 2
            pred_mean = avg_gap
            pred_std = 0.5

        results.append({
            'composite_id': comp_id,
            'material1_id': comp['material1_id'],
            'material2_id': comp['material2_id'],
            'material1_formula': comp['material1_formula'],
            'material2_formula': comp['material2_formula'],
            'material1_bandgap': comp['material1_bandgap'],
            'material2_bandgap': comp['material2_bandgap'],
            'predicted_bandgap': round(pred_mean, 4),
            'prediction_std': round(pred_std, 4),
            'elements': str(comp['elements']),
        })

    results_csv = os.path.join(output_dir, 'composite_predictions.csv')
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"预测完成！")
    print(f"结果保存至: {results_csv}")

    summary = {
        'timestamp': timestamp,
        'n_composites_generated': len(composites_data),
        'output_dir': output_dir,
        'model_dir': model_dir,
        'catalysis_csv': catalysis_csv,
    }
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"运行摘要保存至: {summary_path}")

    if filter_and_save:
        print("\n开始筛选可用材料...")
        try:
            filtered_df, filtered_csv = filter_and_save(results_csv, output_dir)
            print(f"筛选完成，保存于: {filtered_csv}，候选数: {len(filtered_df)}")
        except Exception as e:
            print(f"筛选失败: {e}")

    return output_dir


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='生成复合光催化材料并预测带隙')
    parser.add_argument('--n', type=int, default=100, help='生成的复合物数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--model-dir', type=str, default=None, help='模型目录')
    parser.add_argument('--catalysis-csv', type=str, default=None, help='催化剂CSV文件路径')
    parser.add_argument('--catalysis-cif', type=str, default=None, help='催化剂CIF文件目录')
    parser.add_argument('--disable-cuda', action='store_true', help='禁用CUDA')

    args = parser.parse_args()

    output_dir = generate_and_predict(
        n_composites=args.n,
        seed=args.seed,
        model_dir=args.model_dir,
        catalysis_csv=args.catalysis_csv,
        catalysis_cif=args.catalysis_cif
    )

    if output_dir:
        print(f"\n所有结果已保存到: {output_dir}")