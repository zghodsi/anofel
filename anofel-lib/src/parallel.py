from multiprocessing import Pool
import numpy as np
from config import config

from typing import List
import phe

pool = Pool(processes=config.n_pool)

def aggregate_params_parallel(gradients_of_parties: np.ndarray) -> np.ndarray:
    """
    Take an array of encrypted parameters of models from all partieprime_threshold)
    Return array of mean encrypted params.
    """
    length = len(gradients_of_parties[0])
    gradient_matrix = np.asmatrix(gradients_of_parties)
    transposed = []

    for col in range(length):
        transposed.append(gradient_matrix[:, col])

    result = pool.map(np.mean, transposed, chunksize=300)
    #pool.close()
    #pool.join()
    return result

def decrypt_params_parallel(agg, param: List[phe.EncryptedNumber]) -> List[float]:
    """
    Take encrypted aggregate params.
    Return decrypted params.
    """
    if not config.use_he:
        return param

    decrypted = pool.map(agg.decrypt, param, chunksize=300)
    return decrypted


def encrypt_param_parallel(pubkey, param: List[float]) -> np.ndarray:
    # HE mock
    if not config.use_he:
        return np.array(param)

    return np.array(pool.map(pubkey.encrypt, param, chunksize=300))


def encrypt_add_parallel(grad, noise) -> np.ndarray:
    return pool.starmap(np.add, zip(grad, noise), chunksize=300)
