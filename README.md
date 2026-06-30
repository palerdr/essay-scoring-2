# Learning Agency Lab AES 2.0

Local training is the source of truth. Kaggle is only used to run a fixed offline
inference notebook against an uploaded model artifact Dataset.

## Workflow

1. Train locally with `code/train.py`.
2. Each run writes one flat upload folder:

   ```text
   outputs/<date>/<time>/
     kaggle_dataset/             <- upload/select all files in this folder
       aes2_submission.py
       artifact_manifest.json
       config.json
       metrics.json
       model.safetensors
       run_config.yaml
       submission_config.json
       tokenizer_config.json
       tokenizer.json
       val_predictions.csv
   ```

3. Create a private Kaggle Dataset from the files inside `kaggle_dataset/`.
4. Run `submission/aes2_submission.ipynb` on Kaggle with internet disabled.
   The notebook is intentionally one cell and should not need edits between runs.

Set the Kaggle notebook accelerator to `GPU T4 x2`, not `GPU P100`. The current
Kaggle PyTorch image shown by the notebook error supports CUDA architectures
`sm_70+`; P100 is `sm_60`, so CUDA inference fails with `no kernel image is
available for execution on the device`.

For Hugging Face transformer runs, the model weights inside the artifact are
`model.safetensors`, not a `.pt` file. GRU runs use `model.pt`.

By default, `code/train.py` deletes `best_model.pt` after exporting the flat
artifact folder, so there is not a duplicate checkpoint. To keep the checkpoint
for debugging, override:

```powershell
output.keep_checkpoint=true
```

## Train Examples

DeBERTa v3 small, matching the current best local artifact:

```powershell
uv run python code\train.py `
  model.name=microsoft/deberta-v3-small `
  model.max_length=384 `
  train.batch_size=4 `
  train.epochs=3
```

Debug export smoke test:

```powershell
uv run python code\train.py `
  debug.enabled=true `
  debug.max_steps=1 `
  debug.max_train_rows=16 `
  debug.max_val_rows=8 `
  model.name=microsoft/deberta-v3-small `
  model.max_length=64 `
  train.batch_size=2
```

## Validation Runs

Use cross-validation to compare modeling ideas before spending Kaggle submissions.
CV mode does not create `kaggle_dataset/`; it writes validation artifacts in the
Hydra run folder:

```powershell
uv run python code\train.py `
  train.cv_folds=5 `
  model.name=microsoft/deberta-v3-small `
  model.max_length=384 `
  train.batch_size=4
```

Expected CV outputs:

```text
outputs/<date>/<time>/
  folds.csv
  fold_0_val_predictions.csv
  ...
  fold_metrics.csv
  metrics.json
  oof_predictions.csv
  run_config.yaml
```

To debug only one fold cheaply:

```powershell
uv run python code\train.py `
  debug.enabled=true `
  debug.max_steps=1 `
  train.cv_folds=5 `
  train.fold=0
```

To test head+tail truncation for long essays, keep `max_length` fixed and change
only the truncation strategy:

```powershell
uv run python code\train.py `
  train.cv_folds=5 `
  model.name=microsoft/deberta-v3-small `
  model.max_length=512 `
  model.truncation_strategy=head_tail `
  train.batch_size=4
```

`first` uses the normal tokenizer behavior. `head_tail` keeps the beginning and
end of the essay inside the same token budget, which is useful to test because
high-scoring validation essays were almost always longer than the current context
window.

## Current DeBERTa-Small Artifact

The already-trained DeBERTa-small run has been retrofitted into the standard
flat layout here:

```text
outputs/2026-06-26/14-23-56/kaggle_dataset
```

Its reconstructed validation metrics are saved in `metrics.json`:

```text
rounded QWK:    0.7496038669679224
calibrated QWK: 0.7807877455505503
thresholds:     [1.98, 2.65, 3.49, 4.53, 5.42]
```

Treat these thresholds as single-split calibration, not proof of a better model.
The robust next validation step is 5-fold out-of-fold training: collect raw
predictions for every training row from the fold where that row was held out,
tune thresholds once on all OOF predictions, then report OOF QWK.
