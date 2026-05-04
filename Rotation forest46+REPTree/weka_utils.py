import os
import subprocess
import re
import numpy as np


def write_arff(X, y, path, relation='dataset', attr_prefix='att'):
    X = np.asarray(X)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    n_features = X.shape[1]
    with open(path, 'w', encoding='utf8') as f:
        f.write('@RELATION %s\n\n' % relation)
        for i in range(n_features):
            f.write('@ATTRIBUTE %s%d NUMERIC\n' % (attr_prefix, i))
        f.write('@ATTRIBUTE class NUMERIC\n\n')
        f.write('@DATA\n')
        if y is None:
            for row in X:
                f.write(','.join(map(str, row)) + ',?\n')
        else:
            for row, yy in zip(X, y):
                f.write(','.join(map(str, row)) + ',' + str(yy) + '\n')


def ensure_weka_available(weka_jar_path):
    return os.path.isfile(weka_jar_path)


def train_reptree(weka_jar, train_arff, model_out, java_bin='java'):
    cmd = [java_bin, '-cp', weka_jar, 'weka.classifiers.trees.REPTree', '-t', train_arff, '-d', model_out]
    subprocess.run(cmd, check=True, capture_output=True)


def predict_reptree(weka_jar, model_in, test_arff, java_bin='java'):
    cmd = [java_bin, '-cp', weka_jar, 'weka.classifiers.trees.REPTree', '-l', model_in, '-T', test_arff, '-p', '0']
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    preds = []
    for line in res.stdout.splitlines():
        if re.match(r'^\s*\d+', line):
            parts = line.strip().split()
            if len(parts) >= 3:
                tok = parts[2]
                m = re.search(r'([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)$', tok)
                if m:
                    preds.append(float(m.group(1)))
                else:
                    try:
                        preds.append(float(tok))
                    except Exception:
                        preds.append(np.nan)
    return np.array(preds)
