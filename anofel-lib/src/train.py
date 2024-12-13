import asyncio
import time
from typing import List
from pathlib import Path
from uuid import uuid4

#import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer, SGD, lr_scheduler
#from sklearn.metrics import f1_score
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from config import config
from distro import Party, AggGroup
from model import Model, SimpleRNN
import timeit
import os

from tqdm import tqdm
import numpy as np
from parallel import aggregate_params_parallel, decrypt_params_parallel


from trainutils import adjust_learning_rate, test_model, logprint

print("log filename=", config.accexp_name)

class Trainer:
    """
    Performs learning with hybrid approach.
    Uses asyncio for emulating different parties.
    """
    model: Model
    train_loader: DataLoader
    valid_loader: DataLoader
    aggregator: AggGroup
    parties: List[Party]
    start_time: float = 0
    current_epoch: int = 0
    #train_id: str = f'{config.accexp_name}-{str(uuid4())[:6]}'

    def __init__(self, model: Model, train_loader: DataLoader, valid_loader: DataLoader,
                 train_dataset, valid_dataset, user_groups):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.model = model.to(config.device)
        self.user_groups = user_groups
        self.min_sample_num = None

        if config.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        elif config.optimizer == 'sgd':
            self.optimizer = SGD(self.model.parameters(), lr=config.learning_rate,
                                 momentum=config.momentum, weight_decay=1e-4)
        else:
            print("Optimizer not supported")
            exit()

        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,
                            T_max=config.n_epochs-config.n_warmup_lr_epochs)


        self.configure_system()
        
        if config.run_accexp:
            expfilename = "accexps/"+config.accexp_name+".log"
        else:
            expfilename = "accexps/debug.log"
         

        if config.run_benchmark:
            benchfilename = "benchmarks/"+config.benchmark_name+".log"
        else:
            benchfilename = "benchmarks/debug.log"
           

        self.expfile = open(expfilename, "w")
        self.benchfile = open(benchfilename, "w")

    def get_min_sample(self):
        user_len = [len(self.user_groups[i]) for i in range(config.n_parties)]
        return min(user_len)


    def configure_system(self):
        """
        1. Instantiate the aggregator and parties
        2. Generate private and public keys
        """
        start = timeit.default_timer()
        self.aggregator = AggGroup()
        end = timeit.default_timer()
        print ("AggGroup Setup Time: {:.4f}".format(end-start))

        train_loaders = [DataLoader(Subset(self.train_dataset, list(self.user_groups[i])), 
                                    batch_size=config.batch_size, shuffle=True) 
                                    for i in range(config.n_parties)]

        self.min_sample_num = self.get_min_sample()

        # Init parties with base model
        self.parties = [
                Party(model=self.model, pubkey=self.aggregator.pubkey, train_loader=train_loaders[i])
                for i in range(config.n_parties)
                ]


    def fit(self):
        self.start_time = time.time()

        for epoch in range(1, config.n_epochs + 1):
            self.current_epoch = epoch
            print("epoch ", epoch)

            self.model.train()
            self.optimizer.zero_grad()
            self.fit_clients()
            self.update_models()


            # Test
            if epoch % config.test_every == 0:
                # Update local model for test
                test_model(self.model, self.valid_loader, self.expfile)
                # Plot
                #self.plot()
                # End by loss
                #if self.all_losses[-1] < config.min_loss:
                #    break

        self.expfile.close()
        self.benchfile.close()


    def fit_clients(self):
        for party in self.parties:
            party.fit_model(self.current_epoch)
        avg_loc_t = np.mean([party.stats['loc_t'] for party in self.parties])
        logprint ("Local Train Time: {:.4f}".format(avg_loc_t), self.benchfile)


    def update_models(self):

        encrypted_models = [party.get_gradients() for party in self.parties]
        avg_enc_t = np.mean([party.stats['enc_t'] for party in self.parties])
        logprint ("Param Enc Time: {:.4f}".format(avg_enc_t), self.benchfile)

        # Get mean params
        start = timeit.default_timer()
        #aggregate: np.ndarray = self.aggregator.aggregate_params(encrypted_models)
        aggregate: np.ndarray = aggregate_params_parallel(encrypted_models)
        end = timeit.default_timer()
        logprint ("Param Agg Time: {:.4f}".format(end-start), self.benchfile)

        # Add noise
        start = timeit.default_timer()
        #new_grads = new_grads.to(config.device)
        grads_noised = self.aggregator.add_noise(aggregate, self.min_sample_num)
        end = timeit.default_timer()
        logprint ("Add Noise Time: {:.4f}".format(end-start), self.benchfile)

        # Decrypt
        start = timeit.default_timer()
        #new_grads: List[Tensor] = self.aggregator.decrypt_params(grads_noised)
        new_grads: List[Tensor] = Tensor(decrypt_params_parallel(self.aggregator,grads_noised))
        end = timeit.default_timer()
        logprint ("Param Dec Time: {:.4f}".format(end-start), self.benchfile)

        self.update_grads(new_grads)

        new_model_params: List[Tensor] = []
        for param in self.model.parameters():
            new_model_params.append(param)

        # Update before next epoch
        for party in self.parties:
            party.update_params(new_model_params)



    def update_params(self, new_params: Tensor) -> None:
        """Copy data from new parameters into party's model."""
        with torch.no_grad():
            for model_param, new_param in zip(self.model.parameters(), new_params):
                # Reshape new param and assign into model
                model_param.data = new_param.view_as(model_param.data).to(config.device)

    def update_grads(self, new_grads: Tensor) -> None:
        index=0
        for p in self.model.parameters():
            plen = len(p.data.view(-1).tolist())
            p.grad = new_grads[index:index+plen].view_as(p).to(config.device)
            index+=plen
        self.optimizer.step()


