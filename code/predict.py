from pathlib import Path

import hydra
import pandas as pd
import torch
from data import EssayDataset, ID_COL, load_competition_data
from hydra.core.hydra_config import HydraConfig
from metrics import logits_to_scores
from models import build_model, build_tokenizer
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    
    print("----------------------- Beginning Inference -----------------------------")
    

    _, test_df = load_competition_data(cfg.dataset.competition_handle, cfg.dataset.input_dir)
    tokenizer = build_tokenizer(cfg.model.name)
    test_set = EssayDataset(
        df=test_df,
        tokenizer=tokenizer,
        max_length=cfg.model.max_length,
        truncation_strategy=cfg.model.truncation_strategy,
        has_labels=False,
        task=cfg.model.task,
    )
    test_loader = DataLoader(test_set, batch_size=cfg.predict.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name=cfg.model.name,
        num_labels=cfg.model.num_labels,
        task=cfg.model.task,
        architecture=cfg.model.architecture,
    )
    
    model.load_state_dict(torch.load(cfg.output.checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    scores = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            scores.extend(logits_to_scores(outputs.logits, cfg.model.task))

    submission = pd.DataFrame({ID_COL: test_df[ID_COL], "score": scores})
    output_path = Path(HydraConfig.get().runtime.output_dir) / cfg.output.submission_name
    submission.to_csv(output_path, index=False)
    
    print(f"Predictions written: {output_path}")
    


if __name__ == "__main__":
    main()
