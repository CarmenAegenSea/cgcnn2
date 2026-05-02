import os
import warnings
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Element


def read_id_prop(root_dir):
    """Read id_prop.csv under dataset root.

    Returns a pandas DataFrame with columns ['id', 'target'].
    """
    path = os.path.join(root_dir, "id_prop.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"id_prop.csv not found in {root_dir}")
    df = pd.read_csv(path, header=None, names=["id", "target"], dtype={"id": str})
    df["id"] = df["id"].astype(str).str.strip()
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    return df


def build_features(root_dir, element_list=None, ids=None, verbose=False):
    """Build features for all samples in `root_dir`.

    Features = [elemental fraction vector over `element_list`] + summary stats:
      mean_Z, mean_mass, mean_eneg, std_Z, max_Z, min_Z,
      metal_frac, density, volume_per_atom, n_species, n_atoms

    If `element_list` is None the function first scans all CIFs to discover
    the element set and uses a sorted list (by atomic number) as the ordering.

    Returns: X (ndarray), y (ndarray), ids (list), element_list (list)
    """
    df = read_id_prop(root_dir)
    if ids is None:
        ids = df["id"].tolist()

    # discover elements if needed
    discovered = set()
    if element_list is None:
        for mid in ids:
            cif_path = os.path.join(root_dir, f"{mid}.cif")
            if not os.path.exists(cif_path):
                if verbose:
                    warnings.warn(f"Missing CIF for {mid}: {cif_path}")
                continue
            try:
                s = Structure.from_file(cif_path)
            except Exception as e:
                warnings.warn(f"Failed to read {cif_path}: {e}")
                continue
            for el in s.composition.elements:
                discovered.add(el.symbol)
        element_list = sorted(discovered, key=lambda x: Element(x).Z)
    else:
        element_list = list(element_list)

    elem_idx = {el: i for i, el in enumerate(element_list)}
    features = []
    targets = []
    valid_ids = []

    for mid in ids:
        cif_path = os.path.join(root_dir, f"{mid}.cif")
        if not os.path.exists(cif_path):
            warnings.warn(f"CIF not found for {mid}, skipping: {cif_path}")
            continue
        try:
            s = Structure.from_file(cif_path)
        except Exception as e:
            warnings.warn(f"Failed to read {cif_path}: {e}")
            continue

        comp = s.composition.get_el_amt_dict()
        total = sum(comp.values()) if comp else 0
        if total == 0:
            warnings.warn(f"No composition for {mid}, skipping")
            continue

        frac_vec = np.zeros(len(element_list), dtype=float)
        Z_vals = []
        for el_sym, amt in comp.items():
            frac = amt / total
            if el_sym in elem_idx:
                frac_vec[elem_idx[el_sym]] = frac
            try:
                E = Element(el_sym)
                # extend Z_vals by approximate counts for simple statistics
                Z_vals.extend([E.Z] * int(round(amt)))
            except Exception:
                pass

        # summary statistics
        mean_Z = float(np.mean(Z_vals)) if Z_vals else 0.0
        std_Z = float(np.std(Z_vals)) if Z_vals else 0.0
        max_Z = float(max(Z_vals)) if Z_vals else 0.0
        min_Z = float(min(Z_vals)) if Z_vals else 0.0

        mean_mass = 0.0
        mean_eneg = 0.0
        metal_frac = 0.0
        for el_sym, amt in comp.items():
            frac = amt / total
            try:
                E = Element(el_sym)
                mass = E.atomic_mass or 0.0
                eneg = E.X or 0.0
                is_metal = 1.0 if E.is_metal else 0.0
            except Exception:
                mass = 0.0
                eneg = 0.0
                is_metal = 0.0
            mean_mass += frac * mass
            mean_eneg += frac * eneg
            metal_frac += frac * is_metal

        density = getattr(s, "density", 0.0)
        volume_per_atom = s.volume / s.num_sites if s.num_sites else 0.0
        n_species = len(comp)
        n_atoms = int(total)

        summary = np.array([
            mean_Z,
            mean_mass,
            mean_eneg,
            std_Z,
            max_Z,
            min_Z,
            metal_frac,
            density,
            volume_per_atom,
            n_species,
            n_atoms,
        ], dtype=float)

        x = np.concatenate([frac_vec, summary])
        features.append(x)

        tgt = df.loc[df["id"] == mid, "target"].values
        if len(tgt) == 0:
            warnings.warn(f"No target for {mid}, skipping")
            continue
        targets.append(float(tgt[0]))
        valid_ids.append(mid)

    if len(features) == 0:
        return np.empty((0, len(element_list) + 11)), np.array([]), [], element_list

    X = np.vstack(features)
    y = np.array(targets, dtype=float)
    return X, y, valid_ids, element_list


if __name__ == '__main__':
    # small smoke test when executing the module directly
    import argparse
    parser = argparse.ArgumentParser(description='Build features for a CIF dataset')
    parser.add_argument('datapath', help='Path to dataset root (contains id_prop.csv and .cif files)')
    args = parser.parse_args()
    X, y, ids, elems = build_features(args.datapath, verbose=True)
    print('Built features:', X.shape)
