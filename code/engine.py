import torch, torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import load_competition_data
from omegaconf import DictConfig
from dataset import EssayDataset
from models import build_tokenizer, build_model, BertClassifier
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from features import TEXT_COL, TARGET, LABEL_OFFSET, ID_COL, NUM_LABELS


def train_one_epoch(model, train_loader, optimizer, device, debug_max_steps=None):
    ...

    

def evaluate(model, val_loader, scheduler, device, debug_max_steps=None):
    ...