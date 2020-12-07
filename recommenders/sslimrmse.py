import sys
import time

from utils.official.Recommender_utils import check_matrix
import scipy.sparse as sp
import numpy as np
from recommenders.Cython.SLIM_MSE_Cython_Epoch import train_multiple_epochs
from recommenders.recommender import Recommender
from sklearn.linear_model import ElasticNet

class SSLIMRMSERecommender(Recommender):
    """ SLIM RMSE RECOMMENDATION SYSTEM ALGORITHM
        slim implementation that minimizes the Root Mean Squared Error (RMSE)
        there is an option to add side information to the algorithm """

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True, learning_rate=1e-5, beta=1, epochs=80, l1_reg=3e-5,
                 add_side_info=True):
        super().__init__(URM, ICM, exclude_seen)
        self.learning_rate = learning_rate
        self.beta = beta
        self.epochs = epochs
        self.l1_reg = l1_reg

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

        print("Starting training with {} learning rate, {} l1_reg, {} epochs".format(self.learning_rate, self.l1_reg, self.epochs))
        self.W, loss, samples_per_seconds = train_multiple_epochs(self.A, self.learning_rate, self.epochs, self.l1_reg)
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


class SSLIMElasticNetRecommender(Recommender):

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=False,
                 alpha=0.0005, beta=1.0, l1_ratio=0.029126214, topk=900, positive_only=True):

        super().__init__(URM, ICM,exclude_seen)
        self.alpha = alpha
        self.beta = beta
        self.l1_ratio = l1_ratio
        self.topk = topk
        self.positive_only = positive_only

    def fit(self):

        """ Fits the ElasticNet model """
        self.model = ElasticNet(alpha=self.alpha,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                warm_start=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        # the matrix that has to be learnt
        self.A = check_matrix(self.URM, format='csc', dtype=np.float32)
        self._train()

    def add_side_information(self):

        """ Adds side information to the algorithm, implementing the so called SSLIM """

        if self.beta is not None:
            self._stack(self.ICM.T, self.beta)

    def _train(self, verbose=True):

        """ Trains the ElasticNet model """

        A = self.A

        # we'll construct the W matrix incrementally
        values, rows, columns = [], [], []

        training_start_time = time.time()
        batch_start_time = training_start_time

        # iterates over all tracks in the URM and compute the W column for each of them
        # self.n_tracks
        for item_id in range(self.URM.shape[1]):

            # consider the current column item as the target for the training problem
            y = A[:, item_id].toarray()

            # set to zero the current column in A
            startptr = A.indptr[item_id]
            endptr = A.indptr[item_id+1]

            # save the data of the current column in a temporary variable
            data_t = A.data[startptr:endptr].copy()

            A.data[startptr:endptr] = 0.0

            # fit the ElasticNet model
            self.model.fit(A, y)

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            relevant_items_partition = (-self.model.coef_).argpartition(self.topk)[0:self.topk]
            # - Sort only the relevant items
            relevant_items_partition_sorting = np.argsort(-self.model.coef_[relevant_items_partition])
            # - Get the original item index
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            # keep only non-zero values
            not_zeros_mask = self.model.coef_[ranking] > 0.0
            ranking = ranking[not_zeros_mask]

            values.extend(self.model.coef_[ranking])
            rows.extend(ranking)
            columns.extend([item_id] * len(ranking))

            # finally, replace the original values of the current track column
            A.data[startptr:endptr] = data_t

            if item_id % 1000 == 0 and verbose:
                print(
                    "Processed {} overall ( {:.2f}% ), previous batch in {:.2f} seconds. Columns per second: {:.0f}".format(
                        item_id,
                        100.0 * float(item_id) / self.URM.shape[1],
                        (time.time() - batch_start_time),
                        float(item_id) / (time.time() - training_start_time)))
                sys.stdout.flush()
                sys.stderr.flush()

                batch_start_time = time.time()

        # generate the sparse weight matrix
        self.W = sp.csr_matrix((values, (rows, columns)), shape=(self.URM.shape[1], self.URM.shape[1]), dtype=np.float32)

        self.predicted_URM = self.URM.dot(self.W)

        if verbose:
            print('SLIM RMSE training computed in {:.2f} minutes'.format((time.time() - training_start_time) / 60))

    def _stack(self, to_stack, param, format='csc'):

        """
        Stacks a new sparse matrix under the A matrix used for training
        :param to_stack: sparse matrix to add
        :param param: regularization
        :param format: default 'csc'
        """

        tmp = check_matrix(to_stack, 'csc', dtype=np.float32)
        tmp = tmp.multiply(param)
        self.A = sp.vstack((self.A, tmp), format=format, dtype=np.float32)
