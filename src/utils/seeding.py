"""Process-wide seeding helpers."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed Python and torch for deterministic runs."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
