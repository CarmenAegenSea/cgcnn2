"""Smoke test: verify featurizer and basic pipeline inputs without SHAP/XGBoost.

This script purposely avoids loading the saved model (which may require xgboost
installed). It tests that the featurizer runs and produces expected feature shape.
"""
import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(__file__))
from shap_analysis import basic_check


def main():
    parser = argparse.ArgumentParser(description='SHAP smoke test')
    parser.add_argument('--data', default='XGBoost/clean_catalysis_min2.csv')
    args = parser.parse_args()
    info = basic_check(args.data)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
