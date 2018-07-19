from typing import *

import numpy as np
from tqdm import tqdm


NA = np.newaxis


class BatchGenerator:
    def __init__(self, data: np.array, batch_size: int, shuffle: False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batch_inds_: np.array = None
        self.left_: int = None

        self.initialize()

    def initialize(self):
        data_inds = np.arange(self.data.shape[0])
        np.random.shuffle(data_inds)
        self.batch_inds_ = np.array_split(data_inds, len(self))
        self.left_ = len(self.batch_inds_)

    def __len__(self):
        return self.data.shape[0] // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.left_ == 0:
            self.left_ = len(self.batch_inds_)
            if self.shuffle:
                np.random.shuffle(self.batch_inds_)
            raise StopIteration
        self.left_ -= 1
        return self.data[self.batch_inds_[self.left_]]


class Som:
    def __init__(self, width: int, height: int, depth: int, dist: str= 'cosine'):
        self.width = width
        self.height = height
        self.depth = depth
        self.dist = dist

        self.grid_: np.array = None  # shape=(width, height, 2)
        self.grid_norms_: np.array = None  # shape=(width, height)
        self.weights_: np.array = None  # shape=(width, height, depth)
        self.weights_norms_: np.array = None  # shape=(width, height)
        self.greater_is_closer_: bool = None

        self.initialize()

    def initialize(self):
        self.grid_ = np.asarray([[[i, j] for j in range(self.height)] for i in range(self.width)])
        self.grid_norms_ = (self.grid_ * self.grid_).sum(axis=2) ** 0.5
        # Initialize with small random values
        self.weights_ = np.random.uniform(low=-.1, high=+.1, size=(self.width, self.height, self.depth))
        if self.dist in ['cosine']:
            self.weights_norms_ = (self.weights_ * self.weights_).sum(axis=2) ** 0.5
        if self.dist in ['cosine']:
            self.greater_is_closer_ = True
        elif self.dist in ['euclidean']:
            self.greater_is_closer_ = False
        else:
            raise ValueError(f"Unrecognized similarity={self.dist}")

    def similarity(self, inp: np.array) -> np.array:
        assert inp.ndim == 2
        w, i = self.weights_, inp
        # Similarity of each cell weights with the inputs
        if self.dist == 'cosine':
            w_norm, i_norm = self.weights_norms_, (inp * inp).sum(axis=1) ** 0.5
            s = np.dot(w, i.T) / w_norm[:, :, NA] / i_norm[NA, NA, :]
            s = np.rollaxis(s, axis=2)
        elif self.dist == 'euclidean':
            s = ((i[:, NA, NA, :] - w[NA, :, :, :]) ** 2).sum(axis=3) ** 0.5
        return s  # shape=(batch_size, width, height)

    def closest(self, inp: np.array=None, inp_sim: np.array=None) -> np.array:
        if inp is not None: assert inp.ndim == 2
        if inp_sim is not None: assert inp_sim.ndim == 3
        s = self.similarity(inp=inp) if inp_sim is None else inp_sim
        # Most similar cells for each of the input
        argclose = np.argmax if self.greater_is_closer_ else np.argmin  # shape=(batch_size, 2)
        return np.asarray(list(zip(*np.unravel_index(argclose(s.reshape(s.shape[0], -1), axis=1), s.shape[1:]))))

    def update(self, inp: np.array, lr: float, spread_func: Callable):
        assert inp.ndim == 2
        batch_size = inp.shape[0]
        if self.dist == 'cosine':
            sim = self.similarity(inp=inp)  # shape=(batch_size, width, height)
            winners = self.closest(inp_sim=sim)  # shape=(batch_size, 2)
            spread = spread_func(winners)  # shape=(batch_size, width, height)
            dw = np.dot(np.rollaxis(spread * sim, axis=0, start=3), inp)  # shape=(width, height, depth)
        elif self.dist == 'euclidean':
            dw = inp[:, NA, NA, :] - self.weights_[NA, :, :, :]  # shape=(batch_size, width, height, depth)
            sim = (dw ** 2).sum(axis=3) ** 0.5  # shape=(batch_size, width, height)
            winners = self.closest(inp_sim=sim)  # shape=(batch_size, 2)
            spread = spread_func(winners)  # shape=(batch_size, width, height)
            dw = (spread[:, :, :, NA] * dw).sum(axis=0)  # shape=(width, height, depth)
        self.weights_ += lr * dw / batch_size
        if self.dist in ['cosine']:
            self.weights_norms_ = (self.weights_ * self.weights_).sum(axis=2) ** 0.5


class Neighbourhood:

    def __call__(self, centers: np.array) -> np.array:
        return np.array([])

    def next_iter(self):
        pass

    def next_epoch(self):
        pass


class GaussianNeighbourhood(Neighbourhood):

    def __init__(self, som: Som, sigma_decay: Callable=None):
        self.som = som
        self.sigma_decay = sigma_decay

        self.iter_: int = 0
        self.sigma_: float = self.sigma_decay(0)

    def __call__(self, centers: np.array) -> np.array:
        assert centers.ndim == 2
        m, m2 = self.som.grid_, self.som.grid_norms_ ** 2
        c, c2 = centers, (centers * centers).sum(axis=1)
        # Distance from each cell to the centers
        d2 = m2[:, :, NA] - 2 * np.dot(m, c.T) + c2[NA, NA, :]  # shape=(batch_size, height, width)
        d2 = np.rollaxis(d2, axis=2)
        s2 = self.sigma_ ** 2
        return np.exp(- d2 / (2 * s2))

    def next_iter(self):
        self.iter_ += 1
        self.sigma_ = self.sigma_decay(self.iter_)


class SomTrainer:

    def __init__(self, som: Som, neighbourhood: Neighbourhood, lr_decay: Callable):
        self.som = som
        self.neighbourhood = neighbourhood
        self.lr_decay = lr_decay

        self.iter_: int = 0
        self.lr_: float = self.lr_decay(0)

    def next_iter(self):
        self.iter_ += 1
        self.lr_ = self.lr_decay(self.iter_)
        self.neighbourhood.next_iter()

    def fit(self, batch_gen: Generator, epochs: int, progress_bar=None):
        progress_bar = tqdm if progress_bar is None else progress_bar
        for _ in progress_bar(range(epochs)):
            for batch in batch_gen:
                self.som.update(inp=batch, lr=self.lr_, spread_func=self.neighbourhood)
                self.next_iter()
