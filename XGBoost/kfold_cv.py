"""
XGBoost 5-fold cross-validation on the same split as CGCNN.
Target: band_gap
"""
import os, sys, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

sys.path.append(os.path.dirname(__file__))
from featurizer import ABX3Featurizer

SPLIT_BASE = "data/catalysis_split"
XGB_DATA = "XGBoost/catalysis.csv"
N_FOLDS = 5

# Best params from original model
BEST_PARAMS = {
    "colsample_bytree": 0.8,
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 100,
    "subsample": 1.0,
}

def main():
    # 1. Build fold assignment
    fold_map = {}
    for i in range(1, N_FOLDS + 1):
        path = os.path.join(SPLIT_BASE, str(i), "id_prop.csv")
        df = pd.read_csv(path, header=None, names=["id", "target"])
        for _, row in df.iterrows():
            fold_map[row["id"]] = i

    # 2. Read XGBoost data
    xgb_df = pd.read_csv(XGB_DATA)
    xgb_df["fold"] = xgb_df["material_id"].map(fold_map)

    y_all = xgb_df["band_gap"].values  # target = band_gap

    # 3. Featurize all data once
    featurizer = ABX3Featurizer()
    X_df = featurizer.featurize(xgb_df)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df.fillna(X_df.mean(), inplace=True)

    all_preds = np.array([])
    all_targets = np.array([])

    fold_results = []

    for fold in range(1, N_FOLDS + 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}/{N_FOLDS}")

        train_mask = xgb_df["fold"] != fold
        val_mask = xgb_df["fold"] == fold

        X_train = X_df[train_mask].values
        y_train = y_all[train_mask]
        X_val = X_df[val_mask].values
        y_val = y_all[val_mask]

        # Scale target
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train))
        if y_std < 1e-8:
            y_std = 1.0
        y_train_scaled = (y_train - y_mean) / y_std

        # Pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBRegressor(
                objective="reg:squarederror", random_state=42, n_jobs=1,
                **BEST_PARAMS
            ))
        ])

        pipeline.fit(X_train, y_train_scaled)

        # Predict
        y_pred_scaled = pipeline.predict(X_val)
        y_pred = y_pred_scaled * y_std + y_mean

        # Metrics
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        fold_results.append({"fold": fold, "mae": mae, "rmse": rmse, "r2": r2})
        print(f"  MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

        all_preds = np.concatenate([all_preds, y_pred])
        all_targets = np.concatenate([all_targets, y_val])

    # Overall metrics
    overall_mae = mean_absolute_error(all_targets, all_preds)
    overall_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    overall_r2 = r2_score(all_targets, all_preds)

    print(f"\n{'='*50}")
    print("XGBoost 5-Fold CV Results (target: band_gap)")
    print(f"{'='*50}")
    print(f"Overall MAE:  {overall_mae:.4f} eV")
    print(f"Overall RMSE: {overall_rmse:.4f} eV")
    print(f"Overall R2:   {overall_r2:.4f}")
    print(f"{'='*50}\n")

    # Save results
    results = {
        "overall": {"mae": overall_mae, "rmse": overall_rmse, "r2": overall_r2},
        "fold_results": fold_results,
        "best_params": BEST_PARAMS,
    }
    out_path = os.path.join(os.path.dirname(__file__), "kfold_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_path}")

if __name__ == "__main__":
    main()
