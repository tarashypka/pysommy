from typing import *

import numpy as np
from tqdm import tqdm


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
    def __init__(self, width: int, height: int, depth: int):
        self.width = width
        self.height = height
        self.depth = depth

        self.grid_: np.array = None  # Shape is (width, height, 2)
        self.grid_norms_: np.array = None  # Shape is (width, height)
        self.weights_: np.array = None  # Shape is (width, height, depth)
        self.weights_norms_: np.array = None  # Shape is (width, height)

        self.initialize()

    def initialize(self):
        self.grid_ = np.asarray([[[i, j] for j in range(self.height)] for i in range(self.width)])
        self.grid_norms_ = (self.grid_ * self.grid_).sum(axis=2) ** 0.5
        # Initialize with small random values
        self.weights_ = np.random.uniform(low=-.1, high=+.1, size=(self.width, self.height, self.depth))
        self.weights_norms_ = (self.weights_ * self.weights_).sum(axis=2) ** 0.5
        #self.weights_ /= self.weights_norms_[:, :, np.newaxis]

    def similarity(self, inp: np.array):
        assert inp.ndim == 2
        w, w_norm = self.weights_, self.weights_norms_
        i, i_norm = inp, (inp * inp).sum(axis=1) ** 0.5
        # Similarity of each cell weights with the inputs. Shape is (batch_size, height, width)
        s = np.dot(w, i.T) / w_norm[:, :, np.newaxis] / i_norm[np.newaxis, np.newaxis, :]
        return np.rollaxis(s, axis=2)

    def closest(self, inp: np.array=None, inp_sim: np.array=None) -> np.array:
        if inp is not None: assert inp.ndim == 2
        if inp_sim is not None: assert inp_sim.ndim == 3
        s = self.similarity(inp=inp) if inp_sim is None else inp_sim
        # Most similar cells for each of the input. Shape is (batch_size, 2)
        return np.asarray(list(zip(*np.unravel_index(s.reshape(s.shape[0], -1).argmax(axis=1), s.shape[1:]))))

    def adjust_weights(self, inp: np.array, lr: float, spread_func: Callable):
        assert inp.ndim == 2
        batch_size = inp.shape[0]
        sim = self.similarity(inp=inp)  # Shape is (batch_size, height, weight)
        winners = self.closest(inp_sim=sim)  # Shape is (batch_size, 2)
        self.weights_ += lr * np.dot(np.rollaxis(spread_func(winners) * sim, axis=0, start=3), inp) / batch_size
        self.weights_norms_ = (self.weights_ * self.weights_).sum(axis=2) ** 0.5
        #self.weights_ /= self.weights_norms_[:, :, np.newaxis]


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
        # Distance from each cell to the centers. Shape is (batch_size, height, width)
        d2 = m2[:, :, np.newaxis] - 2 * np.dot(m, c.T) + c2[np.newaxis, np.newaxis, :]
        d2 = np.rollaxis(d2, axis=2)
        s2 = self.sigma_ ** 2
        return np.exp(- d2 / (2 * s2)) #/ (2 * np.pi * s2) ** 0.5

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
                self.som.adjust_weights(inp=batch, lr=self.lr_, spread_func=self.neighbourhood)
                self.next_iter()
