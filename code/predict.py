import hydra
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from data import EssayDataset, ID_COL, load_competition_data
from metrics import logits_to_scores
from models import build_model, build_tokenizer
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path='../conf', config_name="config")
def main(cfg: DictConfig) -> None:
    print("----------------------- Beginning Inference -----------------------------")
    m = cfg.model
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    _, test_df = load_competition_data(
        handle=cfg.dataset.competition_handle,
        input_dir=cfg.dataset.input_dir,
    )
    tokenizer = build_tokenizer(m.name)

    test_ds = EssayDataset(
        df=test_df,
        tokenizer=tokenizer,
        max_length=m.max_length,
        truncation_mode=m.get("truncation_strategy", "first"),
        has_labels=False,
        task=m.task,
    )

    test_loader = DataLoader(test_ds, batch_size=cfg.predict.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name=m.name,
        num_labels=m.num_labels,
        task=m.task,
        architecture=m.architecture,
        tokenizer=tokenizer,
        gru_config=m.gru,
    )
    model.task = m.task
    state_dict = torch.load(cfg.output.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    all_pred_scores = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(device) for k,v in batch.items()}
            outputs = model(**batch)

            pred_scores = logits_to_scores(outputs.logits, m.task)
            all_pred_scores.extend(pred_scores)
        
    submission = pd.DataFrame({
    "essay_id": test_df[ID_COL].tolist(),
    "score": all_pred_scores,
    })
    submission_path = output_dir / cfg.output.submission_name
    submission.to_csv(submission_path, index=False)
    print(f"----------------------- Predictions Written: {submission_path} -----------------------------")

if __name__ == "__main__":
    main()
