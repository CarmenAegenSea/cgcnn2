import re
import numpy as np
import pandas as pd
import sys

try:
    from pymatgen.core import Element
    _HAS_PYMATGEN = True
except Exception:
    Element = None
    _HAS_PYMATGEN = False


class ABX3Featurizer:
    """Featurizer for ABX3 perovskite composition data.

    输入要求（至少一种）：
    - 三列 `A`, `B`, `X`
    - 或者 `formula` 列（例如 `CsPbI3`），脚本会尝试识别 X（计数最大）并把剩余两个元素视为 A/B。
    """

    def __init__(self):
        if not _HAS_PYMATGEN:
            raise ImportError("pymatgen is required for featurization. Install via 'pip install pymatgen' or use the provided requirements.txt")

    @staticmethod
    def parse_formula(formula):
        tokens = re.findall(r'([A-Z][a-z]*)(\d*)', str(formula))
        comp = {}
        for sym, cnt in tokens:
            cnt = int(cnt) if cnt else 1
            comp[sym] = comp.get(sym, 0) + cnt
        return comp

    def _select_ABX_from_formula(self, formula):
        comp = self.parse_formula(formula)
        if not comp:
            return (None, None, None)
        # X is most abundant element (e.g., count 3)
        sorted_items = sorted(comp.items(), key=lambda x: -x[1])
        X_elem = sorted_items[0][0]
        others = [k for k in comp if k != X_elem]
        if len(others) >= 2:
            A_elem = others[0]
            B_elem = others[1]
        elif len(others) == 1:
            A_elem = others[0]
            B_elem = others[0]
        else:
            A_elem = B_elem = X_elem
        return A_elem, B_elem, X_elem

    @staticmethod
    def _safe_getattr(obj, attr, default=None):
        try:
            return getattr(obj, attr)
        except Exception:
            return default

    def _get_element_props(self, symbol):
        if symbol is None:
            return {'Z': None, 'mass': None, 'X': None, 'row': None, 'group': None}
        try:
            el = Element(symbol)
            return {
                'Z': self._safe_getattr(el, 'Z', None),
                'mass': self._safe_getattr(el, 'atomic_mass', None),
                'X': self._safe_getattr(el, 'X', None),
                'row': self._safe_getattr(el, 'row', None),
                'group': self._safe_getattr(el, 'group', None),
            }
        except Exception:
            return {'Z': None, 'mass': None, 'X': None, 'row': None, 'group': None}

    def featurize_row(self, row):
        if all(c in row.index for c in ['A', 'B', 'X']):
            A_sym = row['A']
            B_sym = row['B']
            X_sym = row['X']
        elif 'formula' in row.index:
            A_sym, B_sym, X_sym = self._select_ABX_from_formula(row['formula'])
        else:
            raise ValueError("Data must contain columns 'A','B','X' or 'formula' for featurization")

        A_props = self._get_element_props(A_sym)
        B_props = self._get_element_props(B_sym)
        X_props = self._get_element_props(X_sym)

        features = {}
        props = ['Z', 'mass', 'X', 'row', 'group']
        for p in props:
            a_val = A_props.get(p)
            b_val = B_props.get(p)
            x_val = X_props.get(p)
            features[f'A_{p}'] = a_val
            features[f'B_{p}'] = b_val
            features[f'X_{p}'] = x_val
            try:
                features[f'weighted_{p}'] = ((a_val or 0) + (b_val or 0) + 3 * (x_val or 0)) / 5.0
            except Exception:
                features[f'weighted_{p}'] = None
            try:
                features[f'delta_AB_{p}'] = (a_val or 0) - (b_val or 0)
                features[f'delta_AX_{p}'] = (a_val or 0) - (x_val or 0)
                features[f'delta_BX_{p}'] = (b_val or 0) - (x_val or 0)
            except Exception:
                features[f'delta_AB_{p}'] = None
                features[f'delta_AX_{p}'] = None
                features[f'delta_BX_{p}'] = None

        return features

    def featurize(self, df):
        rows = []
        for idx, row in df.iterrows():
            try:
                feat = self.featurize_row(row)
            except Exception:
                feat = {}
                for p in ['Z', 'mass', 'X', 'row', 'group']:
                    feat[f'A_{p}'] = None
                    feat[f'B_{p}'] = None
                    feat[f'X_{p}'] = None
                    feat[f'weighted_{p}'] = None
                    feat[f'delta_AB_{p}'] = None
                    feat[f'delta_AX_{p}'] = None
                    feat[f'delta_BX_{p}'] = None
            rows.append(feat)
        return pd.DataFrame(rows)
