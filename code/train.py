import json
import random
import hydra
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import ID_COL, TARGET_COL, EssayDataset, load_competition_data
from metrics import (
    apply_thresholds,
    quadratic_weighted_kappa,
    tune_thresholds,
    logits_to_scores,
    labels_to_scores,
)
from models import build_model, build_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iterate_batches(model, loader, optimizer, device, task, train: bool, max_steps=None):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    y_true, y_pred, raw_values = [], [], []
    loss_count = 0

    context = torch.no_grad() if not train else torch.enable_grad()
    with context:
        for step, batch in enumerate(tqdm(loader, desc="train" if train else "validate", leave=False)):
            if max_steps is not None and step >= max_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits

            if task == "regression":
                loss = F.mse_loss(logits.squeeze(-1).float(), labels.float())
            else:
                loss = F.cross_entropy(logits.float(), labels.long())

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            loss_count += 1

            y_true.extend(labels_to_scores(labels, task))
            y_pred.extend(logits_to_scores(logits, task))
            
            if task == "regression":
                raw_values.extend(logits.squeeze(-1).detach().cpu().tolist())
            else:
                raw_values.extend(logits.argmax(dim=-1).detach().cpu().tolist())

    denom = max(loss_count, 1)
    return total_loss / denom, y_true, y_pred, raw_values


def build_loaders(tokenizer, train_df, val_df, cfg, batch_size):
    train_loader = DataLoader(
        EssayDataset(
            train_df,
            tokenizer=tokenizer,
            max_length=cfg.model.max_length,
            truncation_strategy=cfg.model.truncation_strategy,
            has_labels=True,
            task=cfg.model.task,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        EssayDataset(
            val_df,
            tokenizer=tokenizer,
            max_length=cfg.model.max_length,
            truncation_strategy=cfg.model.truncation_strategy,
            has_labels=True,
            task=cfg.model.task,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def write_artifact(model, tokenizer, cfg, run_metrics, artifact_dir: Path):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    model.save_pretrained(artifact_dir)
    tokenizer.save_pretrained(artifact_dir)

    submission_config = {
        "architecture": "transformer",
        "model_name": cfg.model.name,
        "max_length": int(cfg.model.max_length),
        "truncation_strategy": cfg.model.truncation_strategy,
        "num_labels": int(cfg.model.num_labels),
        "task": cfg.model.task,
        "score_min": 1,
        "score_max": 6,
        "thresholds": run_metrics.get("thresholds"),
        "inference_batch_size": int(cfg.predict.batch_size),
    }
    (artifact_dir / "submission_config.json").write_text(json.dumps(submission_config, indent=2), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
    (artifact_dir / "run_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    (artifact_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "kaggle_dataset",
                "competition": cfg.dataset.competition_handle,
                "model_dir": ".",
                "submission_script": "aes2_submission.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    source = Path(__file__).resolve().parents[1] / "submission" / "aes2_submission.py"
    (artifact_dir / "aes2_submission.py").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (artifact_dir / "README.md").write_text(
        "# AES2 Kaggle Submission Artifact\n\nUpload the files in this folder as a Kaggle Dataset and attach to your submission notebook.",
        encoding="utf-8",
    )
    
    print(f"saved flat Kaggle dataset artifact to {artifact_dir.resolve()}")
    


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.train.seed)
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    train_df, test_df = load_competition_data(cfg.dataset.competition_handle, cfg.dataset.input_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Model: {cfg.model.name}")
    print(f"Truncation: {cfg.model.truncation_strategy}, max_length={cfg.model.max_length}")
    print(f"Using device: {device}")
    
    tokenizer = build_tokenizer(cfg.model.name)

    train_df = train_df.reset_index(drop=True).copy()
    train_df["row_idx"] = np.arange(len(train_df))

    # build splits
    split_ids = []
    if cfg.train.cv_folds > 1:
        for fold_idx, (train_indices, val_indices) in enumerate(
            StratifiedKFold(
                n_splits=int(cfg.train.cv_folds),
                shuffle=True,
                random_state=cfg.train.seed,
            ).split(train_df, train_df[TARGET_COL])
        ):
            split_ids.append(
                (
                    fold_idx,
                    train_df.iloc[train_indices].reset_index(drop=True),
                    train_df.iloc[val_indices].reset_index(drop=True),
                )
            )
    else:
        train_split_df, val_split_df = train_test_split(
            train_df,
            test_size=cfg.train.val_size,
            random_state=cfg.train.seed,
            stratify=train_df[TARGET_COL],
        )
        split_ids.append(("holdout", train_split_df.reset_index(drop=True), val_split_df.reset_index(drop=True)))

    if cfg.train.fold is not None:
        split_ids = [item for item in split_ids if str(item[0]) == str(cfg.train.fold)]

    all_val_predictions = []
    fold_summary = []
    final_checkpoint_path = None
    batch_size = cfg.debug.batch_size if cfg.debug.enabled else cfg.train.batch_size
    epochs = cfg.debug.epochs if cfg.debug.enabled else cfg.train.epochs
    max_steps = cfg.debug.max_steps if cfg.debug.enabled else None

    for split_name, train_split_df, val_split_df in split_ids:
        if cfg.debug.enabled:
            train_split_df = train_split_df.head(cfg.debug.max_train_rows)
            val_split_df = val_split_df.head(cfg.debug.max_val_rows)

        
        print(f"{split_name} train shape: {train_split_df.shape}")
        print(f"{split_name} val shape: {val_split_df.shape}")
        print("train score counts:")
        print(f"{train_split_df[TARGET_COL].value_counts().sort_index()}")
        print("val score counts:")
        print(f"{val_split_df[TARGET_COL].value_counts().sort_index()}")
        

        train_loader, val_loader = build_loaders(tokenizer, train_split_df, val_split_df, cfg, batch_size)
        model = build_model(
            model_name=cfg.model.name,
            num_labels=cfg.model.num_labels,
            task=cfg.model.task,
            architecture=cfg.model.architecture,
        )
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scheduler = (
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            if cfg.train.scheduler == "cosine"
            else None
        )

        checkpoint_path = run_dir / "checkpoints" / f"{split_name}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        best_qwk = -1e9
        best_epoch = 0
        for epoch in range(1, epochs + 1):
            train_loss, train_true, train_pred, _ = iterate_batches(
                model,
                train_loader,
                optimizer,
                device,
                cfg.model.task,
                True,
                max_steps=max_steps,
            )
            val_loss, val_true, val_pred, val_raw = iterate_batches(
                model,
                val_loader,
                None,
                device,
                cfg.model.task,
                False,
                max_steps=max_steps,
            )
            val_qwk = quadratic_weighted_kappa(val_true, val_pred)
            train_qwk = quadratic_weighted_kappa(train_true, train_pred)
            print(
                f"{split_name} epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} train_qwk={train_qwk:.4f} "
                f"val_loss={val_loss:.4f} val_qwk={val_qwk:.4f}"
            )
            

            if val_qwk > best_qwk:
                best_qwk = val_qwk
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)
                
                print(f"saved {checkpoint_path} with val_qwk={val_qwk:.4f}")
                

            if scheduler is not None:
                scheduler.step()

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        _, true_scores, pred_scores, raw_scores = iterate_batches(model, val_loader, None, device, cfg.model.task, False, max_steps=max_steps)
        fold_summary.append(
            {
                "split": split_name,
                "best_epoch": best_epoch,
                "val_qwk": float(quadratic_weighted_kappa(true_scores, pred_scores)),
            }
        )
        all_val_predictions.append(
            pd.DataFrame(
                {
                    "row_idx": val_split_df["row_idx"].reset_index(drop=True),
                    ID_COL: val_split_df[ID_COL].reset_index(drop=True),
                    "true_score": true_scores,
                    "pred_raw": raw_scores,
                    "pred_rounded": pred_scores,
                }
            )
        )
        final_checkpoint_path = checkpoint_path

    oof = pd.concat(all_val_predictions, ignore_index=True).sort_values("row_idx").reset_index(drop=True)
    rounded_qwk = quadratic_weighted_kappa(oof["true_score"], oof["pred_rounded"])
    
    print(f"OOF rounded QWK: {rounded_qwk:.4f}")
    

    if cfg.model.task == "regression":
        thresholds, calibrated_qwk = tune_thresholds(oof["pred_raw"], oof["true_score"])
        calibrated = apply_thresholds(np.asarray(oof["pred_raw"]), thresholds)
    else:
        thresholds = None
        calibrated_qwk = rounded_qwk
        calibrated = oof["pred_rounded"]

    oof["pred_calibrated"] = calibrated
    
    print(f"OOF calibrated QWK: {calibrated_qwk:.4f}")
    print(f"Thresholds: {thresholds}")
    

    run_metrics = {
        "model_name": cfg.model.name,
        "architecture": cfg.model.architecture,
        "task": cfg.model.task,
        "seed": int(cfg.train.seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(cfg.train.lr),
        "weight_decay": float(cfg.train.weight_decay),
        "thresholds": thresholds,
        "cv_folds": int(cfg.train.cv_folds),
        "selected_folds": [item[0] for item in split_ids],
        "round_qwk": float(rounded_qwk),
        "calibrated_qwk": float(calibrated_qwk),
    }

    if cfg.train.cv_folds > 1:
        
        print(f"{int(cfg.train.cv_folds)}-FOLD CV SUMMARY")
        print(pd.DataFrame(fold_summary).to_string(index=False))
        print(f"OOF rounded QWK:    {rounded_qwk:.4f}")
        print(f"OOF calibrated QWK: {calibrated_qwk:.4f}")
        print(f"Thresholds:         {thresholds}")
        
        oof.to_csv(run_dir / "oof_predictions.csv", index=False)
        pd.DataFrame(fold_summary).to_csv(run_dir / "fold_metrics.csv", index=False)
        (run_dir / "metrics.json").write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
    else:
        
        print("HOLDOUT SUMMARY")
        print(f"Best epoch:         {fold_summary[0]['best_epoch']}")
        print(f"Rounded QWK:        {rounded_qwk:.4f}")
        print(f"Calibrated QWK:     {calibrated_qwk:.4f}")
        print(f"Thresholds:         {thresholds}")
        
        oof.to_csv(run_dir / "val_predictions.csv", index=False)

        if final_checkpoint_path is not None and final_checkpoint_path.exists():
            final_model = build_model(
                model_name=cfg.model.name,
                num_labels=cfg.model.num_labels,
                task=cfg.model.task,
                architecture=cfg.model.architecture,
            )
            final_model.load_state_dict(torch.load(final_checkpoint_path, map_location=device))
            write_artifact(final_model, tokenizer, cfg, run_metrics, run_dir / cfg.output.artifact_dir)
            if not cfg.output.keep_checkpoint:
                final_checkpoint_path.unlink()

    (run_dir / "run_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    
    print(f"run config written to {run_dir / 'run_config.yaml'}")
    


if __name__ == "__main__":
    main()
