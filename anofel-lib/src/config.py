from dataclasses import dataclass
from typing import List, Tuple

import time
import torch
from torch import Tensor


@dataclass
class Config:
    n_parties: int = 16
    n_aggs: int = 3
    n_agg_groups: int = 1
    threshold: int = 1
    batch_size: int = 32
    key_length: int = 256
    n_epochs: int = 200
    min_loss: float = 0.1
    learning_rate: float = 0.01
    optimizer: str = 'adam'
    momentum: float = 0.9
    n_warmup_lr_epochs: int = 5
    hidden_size: int = 32
    test_every: int = 1
    device: torch.device = torch.device('cuda')
    #device: torch.device = torch.device('cpu')
    use_he: bool = False 
    use_dp: bool = False 
    use_clip: bool = True
    clipping: float = 2
    epsilon: float = 0.9
    delta: float = 0.00001
    sensitivity: float = 1
    use_pool: bool = True 
    n_pool: int = 64
    dataset: str = 'mnist'
    iid: bool = True
    dirichlet: bool = True
    dirichlet_deg: float = 1 
    unequal: bool = False   # sorted non-iid
    datadir: str = 'data/'
    run_accexp: bool = False
    run_benchmark: bool = False 
    run_id: str = str(int(time.time()))
    accexp_name: str = (
        f"{dataset}" +
        f"_n{n_parties}" +
        f"_lr{learning_rate}" +
        f"_{optimizer}" +
        f"_batch{batch_size}" +
        ("_iid" if iid else "_non-iid") +
        ("" if iid else f"_dirichlet{dirichlet_deg}") +
        ("_he" if use_he else "") +
        ("_dp" if use_dp else "") +
        (f"_clip{clipping}" if use_clip else "") +
        (f"_eps{epsilon}_delta{delta}_sens{sensitivity}" if use_dp else "") +
        f"_{run_id}"
    )
    benchmark_name: str = (
        f"{dataset}" +
        f"_n{n_parties}" +
        f"_p{n_pool}" +
        ("_he" if use_he else "") +
        f"_{run_id}"
    )
 

config = Config()

Batch = List[Tuple[Tensor, int]]
