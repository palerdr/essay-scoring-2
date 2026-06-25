import torch
from metrics import cohen_kappa_score, label_to_score

def train_one_epoch(model, train_loader, optimizer, device, debug_max_steps=None):
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for step, batch in enumerate(train_loader):
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

    avg_loss = total_loss / max(len(train_loader), 1)
    qwk = cohen_kappa_score(y_true, y_pred)

    return avg_loss, qwk
        

    

def evaluate(model, val_loader, device, debug_max_steps=None):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
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

    avg_loss = total_loss / max(len(val_loader), 1)
    qwk = cohen_kappa_score(y_true, y_pred)

    return avg_loss, qwk