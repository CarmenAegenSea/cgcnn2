python -m pip install scikit-learn joblib pandas pymatgen matplotlib scipy

# 训练
python train.py data/catalysis/cif --out rf_baseline --target-name band_gap --n-estimators 200 --max-depth 20 --seed 42 --test-size 0.1 --val-size 0.1 --n-jobs -1

# 运行预测
python tree\predict.py tree\smoke_run\model.joblib data\catalysis\cif --csv-output tree\smoke_run\preds.csv