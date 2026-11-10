"""Small utilities shared by the four Kaggle experiments."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def loader(dataset, batch, shuffle=False):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def save_state(path, model, **metadata):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({"state": model.state_dict(), **metadata}, temporary)
    temporary.replace(path)


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
