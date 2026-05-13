"""
MEGNet 5-fold cross-validation on the same split as CGCNN.
Target: band_gap
"""
import os, sys, json, csv, time, warnings, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from random import sample, seed as set_seed
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cgcnn.cgcnn.data import CIFData, collate_pool, get_train_val_test_loader
from MEGNet.model import MEGNet

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

SPLIT_BASE = os.path.abspath("data/catalysis_split")
WORK_BASE = os.path.abspath("data/kfold_temp")
N_FOLDS = 5
MEGNET_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_ARGS = {
    "epochs": 200,
    "batch_size": 256,
    "lr": 0.001,
    "node_dim": 64,
    "edge_dim": 64,
    "hidden_dim": 128,
    "n_blocks": 3,
    "h_fea_len": 128,
    "n_h": 1,
    "loss": "huber",
    "print_freq": 50,
    "cuda": torch.cuda.is_available(),
}


class Normalizer:
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)
        if self.std < 1e-8:
            self.std = torch.tensor(1.0)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def predict(model, loader, normalizer, cuda):
    model.eval()
    all_preds, all_targets, all_ids = [], [], []
    for input, target, batch_ids in loader:
        if cuda:
            input_var = tuple(
                x.cuda(non_blocking=True) if torch.is_tensor(x)
                else [y.cuda(non_blocking=True) for y in x]
                for x in input
            )
        else:
            input_var = input
        output = model(*input_var)
        pred = normalizer.denorm(output.data.cpu())
        all_preds.extend(pred.view(-1).tolist())
        all_targets.extend(target.view(-1).tolist())
        all_ids.extend(batch_ids)
    return np.array(all_ids), np.array(all_targets), np.array(all_preds)


def main():
    all_preds = np.array([])
    all_targets = np.array([])
    fold_results = []

    for fold in range(1, N_FOLDS + 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{N_FOLDS}")
        print(f"{'='*60}")

        train_dir = os.path.join(WORK_BASE, str(fold), "train")
        val_dir = os.path.join(SPLIT_BASE, str(fold))

        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")

        # ---- Data ----
        print("Loading training data ...")
        dataset = CIFData(train_dir, random_seed=42)
        # Use 80/10/10 split to avoid the test_size=0 bug in get_train_val_test_loader
        train_loader, val_loader, _ = get_train_val_test_loader(
            dataset=dataset, collate_fn=collate_pool,
            batch_size=TRAIN_ARGS["batch_size"],
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            num_workers=0, return_test=True,
            train_size=None, val_size=None, test_size=None,
        )
        print(f"  Dataset: {len(dataset)}  Train: {len(train_loader.sampler)}  "
              f"Val: {len(val_loader.sampler)}")

        # ---- Normalizer ----
        n_sample = min(500, len(dataset) if len(dataset) > 0 else 1)
        sample_list = [dataset[i] for i in sample(range(len(dataset)), n_sample)]
        _, sample_target, _ = collate_pool(sample_list)
        normalizer = Normalizer(sample_target)
        print(f"  Target stats: mean={normalizer.mean:.4f}  std={normalizer.std:.4f}")

        # ---- Model ----
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]

        model = MEGNet(
            orig_atom_fea_len=orig_atom_fea_len, nbr_fea_len=nbr_fea_len,
            node_dim=TRAIN_ARGS["node_dim"], edge_dim=TRAIN_ARGS["edge_dim"],
            hidden_dim=TRAIN_ARGS["hidden_dim"], n_blocks=TRAIN_ARGS["n_blocks"],
            h_fea_len=TRAIN_ARGS["h_fea_len"], n_h=TRAIN_ARGS["n_h"],
        )
        if TRAIN_ARGS["cuda"]:
            model.cuda()

        loss_map = {"mse": nn.MSELoss(), "huber": nn.HuberLoss(), "l1": nn.L1Loss()}
        criterion = loss_map[TRAIN_ARGS["loss"]]
        optimizer = optim.Adam(model.parameters(), lr=TRAIN_ARGS["lr"])
        scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.1)

        # ---- Training ----
        best_val_loss = float("inf")
        fold_model_path = os.path.join(MEGNET_DIR, f"fold{fold}_best.pth.tar")
        fold_norm_path = os.path.join(MEGNET_DIR, f"fold{fold}_normalizer.pth")

        for epoch in range(TRAIN_ARGS["epochs"]):
            model.train()
            train_losses = AverageMeter()
            for i, (input, target, _) in enumerate(train_loader):
                if TRAIN_ARGS["cuda"]:
                    input_var = tuple(
                        x.cuda(non_blocking=True) if torch.is_tensor(x)
                        else [y.cuda(non_blocking=True) for y in x]
                        for x in input
                    )
                else:
                    input_var = input
                target_normed = normalizer.norm(target)
                if TRAIN_ARGS["cuda"]:
                    target_var = target_normed.cuda(non_blocking=True)
                else:
                    target_var = target_normed
                output = model(*input_var)
                loss = criterion(output, target_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.update(loss.item(), target.size(0))

            model.eval()
            val_losses = AverageMeter()
            with torch.no_grad():
                for input, target, _ in val_loader:
                    if TRAIN_ARGS["cuda"]:
                        input_var = tuple(
                            x.cuda(non_blocking=True) if torch.is_tensor(x)
                            else [y.cuda(non_blocking=True) for y in x]
                            for x in input
                        )
                    else:
                        input_var = input
                    target_normed = normalizer.norm(target)
                    if TRAIN_ARGS["cuda"]:
                        target_var = target_normed.cuda(non_blocking=True)
                    else:
                        target_var = target_normed
                    output = model(*input_var)
                    loss = criterion(output, target_var)
                    val_losses.update(loss.item(), target.size(0))

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1:3d}/{TRAIN_ARGS['epochs']}]  "
                      f"Train Loss: {train_losses.avg:.4f}  Val Loss: {val_losses.avg:.4f}")

            if val_losses.avg < best_val_loss:
                best_val_loss = val_losses.avg
                torch.save(model.state_dict(), fold_model_path)
                torch.save(normalizer.state_dict(), fold_norm_path)

            scheduler.step()

        print(f"  Best val loss: {best_val_loss:.4f}")

        # ---- Predict on held-out fold ----
        print("  Predicting on held-out fold ...")
        model.load_state_dict(torch.load(fold_model_path, map_location="cpu"))
        if TRAIN_ARGS["cuda"]:
            model.cuda()
        norm_state = torch.load(fold_norm_path, map_location="cpu")
        normalizer.load_state_dict(norm_state)

        val_dataset = CIFData(val_dir, random_seed=42)
        val_loader_full = DataLoader(
            val_dataset, batch_size=TRAIN_ARGS["batch_size"],
            shuffle=False, collate_fn=collate_pool, num_workers=0,
        )

        ids, targets, preds = predict(model, val_loader_full, normalizer, TRAIN_ARGS["cuda"])

        fold_mae = mean_absolute_error(targets, preds)
        fold_rmse = np.sqrt(mean_squared_error(targets, preds))
        fold_r2 = r2_score(targets, preds)
        fold_results.append({"fold": fold, "mae": fold_mae, "rmse": fold_rmse, "r2": fold_r2})
        print(f"  Fold {fold}: MAE={fold_mae:.4f}, RMSE={fold_rmse:.4f}, R2={fold_r2:.4f}")

        all_targets = np.concatenate([all_targets, targets]) if all_targets.size else targets
        all_preds = np.concatenate([all_preds, preds]) if all_preds.size else preds

        # Clean up fold-specific files
        for p in [fold_model_path, fold_norm_path]:
            if os.path.exists(p):
                os.remove(p)

    # ---- Overall ----
    overall_mae = mean_absolute_error(all_targets, all_preds)
    overall_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    overall_r2 = r2_score(all_targets, all_preds)

    print(f"\n{'='*50}")
    print("MEGNet 5-Fold CV Results (target: band_gap)")
    print(f"{'='*50}")
    print(f"Overall MAE:  {overall_mae:.4f} eV")
    print(f"Overall RMSE: {overall_rmse:.4f} eV")
    print(f"Overall R2:   {overall_r2:.4f}")
    print(f"{'='*50}")

    results = {
        "overall": {"mae": overall_mae, "rmse": overall_rmse, "r2": overall_r2},
        "fold_results": fold_results,
    }
    out_path = os.path.join(MEGNET_DIR, "kfold_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
