import json
import random
import shutil
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ID_COL, TARGET_COL, EssayDataset, add_folds, load_competition_data, train_val_split
from metrics import (
    SCORE_MAX,
    SCORE_MIN,
    apply_thresholds,
    labels_to_scores,
    logits_to_scores,
    quadratic_weighted_kappa,
    tune_thresholds,
)
from models import build_model, build_tokenizer


def set_seed(seed: int) -> None:
    """Make splits and model initialization repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(logits, labels, task: str):
    """Use MSE for regression and cross-entropy for classification."""
    if task == "regression":
        return F.mse_loss(logits.squeeze(-1).float(), labels.float())
    return F.cross_entropy(logits.float(), labels.long())


def train_one_epoch(model, loader, optimizer, device, max_steps=None) -> tuple[float, float]:
    """Run one training epoch and return average loss and QWK."""
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_steps is not None and step >= max_steps:
            break

        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")

        optimizer.zero_grad()
        logits = model(**batch).logits
        loss = compute_loss(logits, labels, model.task)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        y_true.extend(labels_to_scores(labels, model.task))
        y_pred.extend(logits_to_scores(logits, model.task))

    steps = min(len(loader), max_steps) if max_steps is not None else len(loader)
    return total_loss / max(steps, 1), quadratic_weighted_kappa(y_true, y_pred)


def validate(model, loader, device, max_steps=None):
    """Run validation and return loss plus true/rounded/raw predictions."""
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    raw_values = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="val", leave=False)):
            if max_steps is not None and step >= max_steps:
                break

            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits

            total_loss += compute_loss(logits, labels, model.task).item()
            y_true.extend(labels_to_scores(labels, model.task))
            y_pred.extend(logits_to_scores(logits, model.task))
            if model.task == "regression":
                raw_values.extend(logits.squeeze(-1).detach().cpu().tolist())
            else:
                raw_values.extend(logits.argmax(dim=-1).detach().cpu().tolist())

    steps = min(len(loader), max_steps) if max_steps is not None else len(loader)
    return total_loss / max(steps, 1), y_true, y_pred, raw_values


def write_submission_artifact(model, tokenizer, cfg, artifact_dir: Path, metrics: dict) -> None:
    """Write the flat folder uploaded as the Kaggle model Dataset."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    model.eval()

    if cfg.model.architecture == "transformer":
        model.save_pretrained(artifact_dir)
    elif cfg.model.architecture == "gru":
        torch.save(model.state_dict(), artifact_dir / "model.pt")
        gru_config = {
            "vocab_size": len(tokenizer),
            "num_labels": cfg.model.num_labels,
            "pad_token_id": tokenizer.pad_token_id,
            "embedding_dim": cfg.model.gru.embedding_dim,
            "hidden_size": cfg.model.gru.hidden_size,
            "num_layers": cfg.model.gru.num_layers,
            "bidirectional": cfg.model.gru.bidirectional,
            "dropout": cfg.model.gru.dropout,
            "task": cfg.model.task,
        }
        (artifact_dir / "gru_config.json").write_text(json.dumps(gru_config, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unknown model architecture: {cfg.model.architecture}")

    tokenizer.save_pretrained(artifact_dir)
    submission_config = {
        "architecture": cfg.model.architecture,
        "model_name": cfg.model.name,
        "max_length": cfg.model.max_length,
        "truncation_strategy": cfg.model.truncation_strategy,
        "num_labels": cfg.model.num_labels,
        "task": cfg.model.task,
        "score_min": SCORE_MIN,
        "score_max": SCORE_MAX,
        "thresholds": metrics.get("thresholds"),
        "inference_batch_size": cfg.predict.batch_size,
    }
    (artifact_dir / "submission_config.json").write_text(json.dumps(submission_config, indent=2), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (artifact_dir / "run_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    (artifact_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "kaggle_dataset",
                "competition": cfg.dataset.competition_handle,
                "model_dir": ".",
                "submission_script": "aes2_submission.py",
                "metrics_file": "metrics.json",
                "config_file": "submission_config.json",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    script_src = Path(__file__).resolve().parents[1] / "submission" / "aes2_submission.py"
    shutil.copy2(script_src, artifact_dir / "aes2_submission.py")
    (artifact_dir / "README.md").write_text(
        "# AES2 Kaggle submission artifact\n\n"
        "Upload the files in this folder as a private Kaggle Dataset and attach it to the fixed submission notebook.\n",
        encoding="utf-8",
    )
    print(f"saved flat Kaggle dataset artifact to {artifact_dir.resolve()}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    set_seed(cfg.train.seed)

    train_df, test_df = load_competition_data(cfg.dataset.competition_handle, cfg.dataset.input_dir)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Model: {cfg.model.name}")
    print(f"Truncation: {cfg.model.truncation_strategy}, max_length={cfg.model.max_length}")

    tokenizer = build_tokenizer(cfg.model.name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = cfg.debug.batch_size if cfg.debug.enabled else cfg.train.batch_size
    epochs = cfg.debug.epochs if cfg.debug.enabled else cfg.train.epochs
    max_steps = cfg.debug.max_steps if cfg.debug.enabled else None

    cv_folds = int(cfg.train.cv_folds)
    if cv_folds > 1:
        folded = add_folds(train_df, n_folds=cv_folds, seed=cfg.train.seed)
        folded[[ID_COL, TARGET_COL, "fold"]].to_csv(run_dir / "folds.csv", index=False)
        selected_folds = list(range(cv_folds)) if cfg.train.fold is None else [int(cfg.train.fold)]
        split_plan = [
            (
                f"fold_{fold}",
                folded[folded["fold"] != fold].reset_index(drop=True),
                folded[folded["fold"] == fold].reset_index(drop=True),
                run_dir / "checkpoints" / f"fold_{fold}.pt",
            )
            for fold in selected_folds
        ]
    else:
        train_split, val_split = train_val_split(train_df, val_size=cfg.train.val_size, seed=cfg.train.seed)
        if cfg.debug.enabled:
            train_split = train_split.head(cfg.debug.max_train_rows).reset_index(drop=True)
            val_split = val_split.head(cfg.debug.max_val_rows).reset_index(drop=True)
        selected_folds = []
        split_plan = [("holdout", train_split, val_split, run_dir / cfg.output.checkpoint_path)]

    all_predictions = []
    split_metrics = []
    final_model = None
    final_checkpoint = None

    for split_name, split_train_df, split_val_df, checkpoint_path in split_plan:
        print(f"{split_name} train shape: {split_train_df.shape}")
        print(f"{split_name} val shape: {split_val_df.shape}")
        print(f"{split_name} train score counts:\n{split_train_df[TARGET_COL].value_counts().sort_index()}")
        print(f"{split_name} val score counts:\n{split_val_df[TARGET_COL].value_counts().sort_index()}")

        train_dataset = EssayDataset(
            split_train_df,
            tokenizer,
            max_length=cfg.model.max_length,
            truncation_mode=cfg.model.truncation_strategy,
            has_labels=True,
            task=cfg.model.task,
        )
        val_dataset = EssayDataset(
            split_val_df,
            tokenizer,
            max_length=cfg.model.max_length,
            truncation_mode=cfg.model.truncation_strategy,
            has_labels=True,
            task=cfg.model.task,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = build_model(
            model_name=cfg.model.name,
            num_labels=cfg.model.num_labels,
            task=cfg.model.task,
            architecture=cfg.model.architecture,
            tokenizer=tokenizer,
            gru_config=cfg.model.gru,
        )
        model.task = cfg.model.task
        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scheduler = (
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            if cfg.train.scheduler == "cosine"
            else None
        )

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        best_epoch = None
        best_val_loss = None
        best_val_qwk = -1e8

        for epoch in tqdm(range(epochs), desc=split_name):
            train_loss, train_qwk = train_one_epoch(model, train_loader, optimizer, device, max_steps=max_steps)
            val_loss, y_true, y_pred, _ = validate(model, val_loader, device, max_steps=max_steps)
            val_qwk = quadratic_weighted_kappa(y_true, y_pred)
            print(
                f"{split_name} epoch {epoch + 1}/{epochs} "
                f"train_loss={train_loss:.4f} train_qwk={train_qwk:.4f} "
                f"val_loss={val_loss:.4f} val_qwk={val_qwk:.4f}"
            )

            if val_qwk > best_val_qwk:
                best_epoch = epoch + 1
                best_val_loss = float(val_loss)
                best_val_qwk = float(val_qwk)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"saved {checkpoint_path} with val_qwk={val_qwk:.4f}")

            if scheduler is not None:
                scheduler.step()

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        _, y_true, y_pred, raw_values = validate(model, val_loader, device, max_steps=max_steps)

        val_rows = split_val_df.iloc[: len(y_true)].reset_index(drop=True)
        predictions = pd.DataFrame(
            {
                "row_idx": val_rows["row_idx"].tolist() if "row_idx" in val_rows else list(range(len(val_rows))),
                "fold": val_rows["fold"].tolist() if "fold" in val_rows else [-1] * len(val_rows),
                ID_COL: val_rows[ID_COL].tolist(),
                "true_score": y_true,
                "pred_raw": raw_values,
                "pred_rounded": y_pred,
            }
        )
        predictions.to_csv(run_dir / f"{split_name}_val_predictions.csv", index=False)
        all_predictions.append(predictions)

        split_metrics.append(
            {
                "split": split_name,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_qwk_rounded_during_training": best_val_qwk,
                "val_qwk_rounded": float(quadratic_weighted_kappa(y_true, y_pred)),
                "val_prediction_count": int(len(predictions)),
                "checkpoint_path": str(checkpoint_path),
            }
        )

        final_model = model
        final_checkpoint = checkpoint_path
        if cv_folds > 1:
            if not cfg.output.keep_checkpoint and checkpoint_path.exists():
                checkpoint_path.unlink()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    predictions = pd.concat(all_predictions, ignore_index=True).sort_values("row_idx").reset_index(drop=True)
    rounded_qwk = quadratic_weighted_kappa(predictions["true_score"], predictions["pred_rounded"])
    if cfg.model.task == "regression":
        thresholds, calibrated_qwk = tune_thresholds(predictions["pred_raw"], predictions["true_score"])
        predictions["pred_calibrated"] = apply_thresholds(predictions["pred_raw"], thresholds)
    else:
        thresholds = None
        calibrated_qwk = rounded_qwk
        predictions["pred_calibrated"] = predictions["pred_rounded"]

    run_metrics = {
        "model_name": cfg.model.name,
        "architecture": cfg.model.architecture,
        "task": cfg.model.task,
        "max_length": int(cfg.model.max_length),
        "truncation_strategy": cfg.model.truncation_strategy,
        "seed": int(cfg.train.seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(cfg.train.lr),
        "weight_decay": float(cfg.train.weight_decay),
        "scheduler": cfg.train.scheduler,
        "debug": bool(cfg.debug.enabled),
        "thresholds": thresholds,
    }

    if cv_folds > 1:
        predictions.to_csv(run_dir / "oof_predictions.csv", index=False)
        pd.DataFrame(split_metrics).to_csv(run_dir / "fold_metrics.csv", index=False)
        run_metrics.update(
            {
                "validation_mode": "cross_validation",
                "cv_folds": cv_folds,
                "selected_folds": selected_folds,
                "fold_metrics": split_metrics,
                "oof_qwk_rounded": float(rounded_qwk),
                "oof_qwk_calibrated": float(calibrated_qwk),
                "oof_prediction_count": int(len(predictions)),
            }
        )
        (run_dir / "metrics.json").write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
        (run_dir / "run_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
        print("\n5-FOLD CV SUMMARY")
        print(pd.DataFrame(split_metrics)[["split", "best_epoch", "val_qwk_rounded", "best_val_loss"]].to_string(index=False))
        print(f"OOF rounded QWK:    {rounded_qwk:.4f}")
        print(f"OOF calibrated QWK: {calibrated_qwk:.4f}")
        print(f"Thresholds:         {thresholds}")
    else:
        artifact_dir = run_dir / cfg.output.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(artifact_dir / "val_predictions.csv", index=False)
        run_metrics.update(
            {
                "validation_mode": "holdout",
                "val_size": float(cfg.train.val_size),
                "best_epoch": split_metrics[0]["best_epoch"],
                "best_val_loss": split_metrics[0]["best_val_loss"],
                "best_val_qwk_rounded_during_training": split_metrics[0]["best_val_qwk_rounded_during_training"],
                "val_qwk_rounded": float(rounded_qwk),
                "val_qwk_calibrated": float(calibrated_qwk),
                "val_prediction_count": int(len(predictions)),
                "checkpoint_path": str(final_checkpoint),
            }
        )
        print("\nHOLDOUT SUMMARY")
        print(f"Best epoch:         {split_metrics[0]['best_epoch']}")
        print(f"Rounded QWK:        {rounded_qwk:.4f}")
        print(f"Calibrated QWK:     {calibrated_qwk:.4f}")
        print(f"Thresholds:         {thresholds}")
        write_submission_artifact(final_model, tokenizer, cfg, artifact_dir, run_metrics)

        if not cfg.output.keep_checkpoint and final_checkpoint.exists():
            final_checkpoint.unlink()
            print(f"removed checkpoint {final_checkpoint.resolve()}")


if __name__ == "__main__":
    main()
