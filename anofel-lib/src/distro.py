# Use separate multiprocessing library because mapped functions are methods,
# that are not supported with a default library.
import copy
import random
from functools import partial
from multiprocess import Pool, cpu_count
from typing import Callable, Iterable, List, Tuple

#import diffprivlib as dp
import numpy as np
import phe
import torch
from torch import Tensor
from torch.nn import Parameter
from torch.functional import F
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, Optimizer, SGD, lr_scheduler

from distro_paillier import distributed_paillier
from distro_paillier.distributed_paillier import generate_shared_paillier_key
#import distributed_paillier
#from distributed_paillier import generate_shared_paillier_key

from parallel import encrypt_param_parallel, encrypt_add_parallel

from trainutils import adjust_learning_rate
import timeit
from tqdm import tqdm

from config import config
from model import Model


n_cpus = cpu_count()
print ("cpus=", n_cpus)

#pool = Pool(processes=n_cpus - 3)
pool = Pool(processes=config.n_pool)
EncryptedParameter = np.ndarray  # [phe.EncryptedNumber]

#use_pool = True


class AggGroup:
    """Private key holder. Decrypts the average gradient"""

    def __init__(self):
        if not config.use_he:
            # pubkey mock
            self.pubkey = phe.PaillierPublicKey(1)
            return

        Key, _, _, _, PublicKey, _, _, SecretKeyShares, theta = generate_shared_paillier_key(
            keyLength=config.key_length,
            n=config.n_aggs,
            t=config.threshold,
        )

        self.prikey = Key
        self.pubkey = PublicKey

        # decrypt takes one argument -- ciphertext to decode
        self.decrypt = partial(
            Key.decrypt,
            n=config.n_aggs,
            t=config.threshold,
            PublicKey=PublicKey,
            SecretKeyShares=SecretKeyShares,
            theta=theta
        )

        self.mean = partial(
            np.mean,
        )

    def aggregate_params(self, gradients_of_parties: np.ndarray) -> np.ndarray:
        """
        Take an array of encrypted parameters of models from all partieprime_threshold)
        Return array of mean encrypted params.
        """
        return np.mean(gradients_of_parties, axis=0)
        # This parallelization incurs slowdowns
        #if config.use_pool:
        #    length = len(gradients_of_parties[0])
        #    gradient_matrix = np.asmatrix(gradients_of_parties)
        #    transposed = []

        #    for col in range(length):
        #        transposed.append(gradient_matrix[:, col])

        #    result = pool.map(np.mean, transposed, chunksize=300)
        #    #pool.close()
        #    #pool.join()
        #    return result
        #else:
        #    return np.mean(gradients_of_parties, axis=0)


    def decrypt_params(self, param: List[phe.EncryptedNumber]) -> List[float]:
        """
        Take encrypted aggregate params.
        Return decrypted params.
        """
        if not config.use_he:
            return Tensor(param)

        if config.use_pool:
            decrypted = pool.map(self.decrypt, param, chunksize=300)
        else:
            decrypted = [self.decrypt(num) for num in param]
        return Tensor(decrypted)

    def encrypt_noise(self, noise: List[float]) -> EncryptedParameter:
        # HE mock
        if not config.use_he:
            return np.array(noise)

        encrypt = partial(self.pubkey.encrypt)
        if config.use_pool:
            return np.array(pool.map(encrypt, noise, chunksize=300))
        else:
            return np.array([encrypt(num) for num in noise])



    def GaussianNoise(self, epsilon, delta, clipping, min_sample_num, T, size):
        sensitivity = 2*clipping / min_sample_num
        noise_std = (np.sqrt(2*np.log(1.25 / float(delta))) * float(sensitivity) * float(T))  / float(epsilon)
        #return torch.normal(mean=0, std=noise_std, size=size) #.to(config.device)
        return np.random.normal(loc=0, scale=noise_std, size=size)


    def add_noise(self, grad, min_sample_num) -> Tensor:
        """
        Add noise from diffential privacy mechanism.
        param: 1-D (flattened) Parameter
        return Tensor of param's data with applied DP.
        """
        # DP mock
        if not config.use_dp:
            return grad

        noise = self.GaussianNoise(config.epsilon, config.delta, config.clipping, 
                                   min_sample_num, config.n_epochs, len(grad))
                                   #min_sample_num, config.n_epochs, grad.shape)
        
        #encrypted_noise: EncryptedParameter = self.encrypt_noise(noise)
        encrypted_noise: EncryptedParameter = encrypt_param_parallel(self.pubkey, noise)
        #return grad+encrypted_noise
        return encrypt_add_parallel(grad, encrypted_noise)





class Party:
    """
    Using public key can encrypt locally trained model.
    """
    optimizer: torch.optim.Optimizer
    model: Model
    train_loader: DataLoader
    pubkey: phe.PaillierPublicKey
    #randomiser: dp.mechanisms.Gaussian

    def __init__(self, pubkey: phe.PaillierPublicKey, model: Model, train_loader: DataLoader):
        self.model: Model = copy.deepcopy(model).to(config.device)
        self.train_loader = train_loader
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

        self.pubkey = pubkey
        self.stats = {'enc_t': None, 'loc_t': None}
        #self.randomiser = dp.mechanisms.Gaussian().set_epsilon_delta(1, 1).set_sensitivity(0.1)

    def get_gradients(self) -> List[EncryptedParameter]:
        encrypted_params: List[np.ndarray] = []
        encrypted_zeros: List[np.ndarray] = []
        flattened_params = np.empty(0)

        for param in self.model.parameters():
            grad = param.grad
            flattened = grad.data.view(-1)

            flattened = flattened.tolist()
            flattened_params = np.append(flattened_params, np.array(flattened))

        start = timeit.default_timer()
        #encrypted: EncryptedParameter = self.encrypt_param(flattened_params)
        encrypted: EncryptedParameter = encrypt_param_parallel(self.pubkey, flattened_params)

        ## encrypt zeros for other aggregator groups
        #zeros = [0] * len(flattened_params)
        #for j in range(config.n_agg_groups-1):
        #    encrypted_zeros.append(self.encrypt_param(zeros))

        end = timeit.default_timer()
        self.stats['enc_t'] = end-start

        ## write all ciphertexts to file
        #ctfile = open("ciphertexts.txt", "w") 
        #for i in range(len(encrypted)):
        #    ctfile.write(str(encrypted[i].ciphertext()))
        #for i in range(len(encrypted_zeros)):
        #    for j in range(len(encrypted_zeros[i])):
        #        ctfile.write(str(encrypted_zeros[i][j].ciphertext()))
        #ctfile.close()

        return encrypted


    def fit_model(self, current_epoch):
        num_steps_per_epoch = len(self.train_loader)
        start = timeit.default_timer()
        for step, (features, target, indices) in enumerate(tqdm(self.train_loader,
                                desc='train', ncols=0, disable=False)):

            adjust_learning_rate(self.scheduler, current_epoch, step, num_steps_per_epoch,
                         warmup_lr_epochs=config.n_warmup_lr_epochs,
                         schedule_lr_per_epoch=True,
                         size=config.n_parties)

            self.training_step(features, target)

        end = timeit.default_timer()
        self.stats['loc_t'] = end-start


    #def training_step(self, batch: Tuple[Tensor, Tensor], epoch, step, num_steps_per_epoch) -> List[Parameter]:
    def training_step(self, features: Tensor, target: Tensor) -> List[Parameter]:
        """Forward and backward pass"""
        #adjust_learning_rate(self.scheduler, epoch, step, num_steps_per_epoch,
        #                 warmup_lr_epochs=config.n_warmup_lr_epochs,
        #                 schedule_lr_per_epoch=True)

        #features, target = batch
        features, target = features.to(config.device), target.to(config.device)
        self.optimizer.zero_grad()

        pred = self.model(features)

#        loss: Tensor = F.nll_loss(pred, target)
        loss: Tensor = F.cross_entropy(pred, target)

        loss.backward()

        if config.use_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clipping)

        self.optimizer.step()


    def encrypt_param(self, param: List[float]) -> EncryptedParameter:
        # HE mock
        if not config.use_he:
            return np.array(param)

        encrypt = partial(self.pubkey.encrypt)
        if config.use_pool:
            return np.array(pool.map(encrypt, param, chunksize=300))
        else:
            return np.array([encrypt(num) for num in param])

    def update_params(self, new_params: Tensor) -> None:
        """Copy data from new parameters into party's model."""
        with torch.no_grad():
            for model_param, new_param in zip(self.model.parameters(), new_params):
                # Reshape new param and assign into model
                model_param.data = new_param.view_as(model_param.data).to(config.device)

