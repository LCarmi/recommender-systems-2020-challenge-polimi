import sys
import time

from utils.official.Recommender_utils import check_matrix
import scipy.sparse as sp
import numpy as np
from recommenders.Cython.SLIM_MSE_Cython_Epoch import train_multiple_epochs
from recommenders.recommender import Recommender


class SSLIMRMSERecommender(Recommender):
    """ SLIM RMSE RECOMMENDATION SYSTEM ALGORITHM
        slim implementation that minimizes the Root Mean Squared Error (RMSE)
        there is an option to add side information to the algorithm """

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True, learning_rate=1e-3, beta=1.0, epochs=2,
                 add_side_info=True):
        super().__init__(URM, ICM, exclude_seen)
        self.learning_rate = learning_rate
        self.beta = beta
        self.epochs = epochs

        self.A = self.URM.copy()

        if add_side_info:
            self.add_side_information()

    def fit(self):

        self._train()

    def add_side_information(self):

        """ Adds side information to the algorithm, implementing the so called SSLIM """

        if self.beta is not None:
            self._stack(self.ICM.T, self.beta)

    def _train(self, verbose=True):
        self.A = check_matrix(self.A, 'csr', np.float64)

        self.W, loss, samples_per_seconds = train_multiple_epochs(self.A, self.learning_rate, self.epochs)
        self.W = sp.csr_matrix(self.W)
        self.predicted_URM = self.A.dot(self.W)

    def _stack(self, to_stack, param, format='csr'):

        """
        Stacks a new sparse matrix under the A matrix used for training
        :param to_stack: sparse matrix to add
        :param param: regularization
        :param format: default 'csc'
        """

        tmp = check_matrix(to_stack, 'csc', dtype=np.float64)
        tmp = tmp.multiply(param)
        self.A = sp.vstack((self.A, tmp), format=format, dtype=np.float64)
