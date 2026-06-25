import torch
from tqdm import tqdm
from metrics import quadratic_weighted_kappa, label_to_score

def train_one_epoch(model, train_loader, optimizer, device, debug_max_steps=None):
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(train_loader, desc="train", leave=False)):
        if debug_max_steps is not None and step >= debug_max_steps:
            break

        batch = {k: v.to(device) for k,v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
        labels = batch["labels"].detach().cpu().tolist()

        y_pred.extend([label_to_score(p) for p in preds])
        y_true.extend([label_to_score(t) for t in labels])

    steps = min(len(train_loader), debug_max_steps) if debug_max_steps is not None else len(train_loader)
    avg_loss = total_loss / max(steps, 1)
    qwk = quadratic_weighted_kappa(y_true, y_pred)

    return avg_loss, qwk
        

    

def evaluate(model, val_loader, device, debug_max_steps=None):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader, desc="val", leave=False)):
            if debug_max_steps is not None and step >= debug_max_steps:
                break

            batch = {k: v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            preds = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
            labels = batch["labels"].detach().cpu().tolist()

            y_pred.extend([label_to_score(p) for p in preds])
            y_true.extend([label_to_score(t) for t in labels])

    steps = min(len(val_loader), debug_max_steps) if debug_max_steps is not None else len(val_loader)
    avg_loss = total_loss / max(steps, 1)
    qwk = quadratic_weighted_kappa(y_true, y_pred)

    return avg_loss, qwk
