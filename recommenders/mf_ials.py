import numpy as np
import time
import scipy.sparse as sp

from recommenders.recommender import MatrixFactorizationRecommender


def non_zeros(m, row):
    """ Returns an tuple iterator (tracks, data) """

    for index in range(m.indptr[row], m.indptr[row + 1]):
        yield m.indices[index], m.data[index]


def least_squares_cg(Cui, X, Y, lambda_val, cg_steps=3):
    """
    Computes the least square solution using the conjugate gradient
    Check: http://www.benfrederickson.com/fast-implicit-matrix-factorization/
    """

    users, features = X.shape

    YtY = Y.T.dot(Y) + lambda_val * np.eye(features)

    for u in range(users):

        x = X[u]
        r = -YtY.dot(x)

        for i, confidence in non_zeros(Cui, u):
            r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]

        p = r.copy()
        rsold = r.dot(r)

        for it in range(cg_steps):
            Ap = YtY.dot(p)
            for i, confidence in non_zeros(Cui, u):
                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap

            rsnew = r.dot(r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x


class ALSMFRecommender(MatrixFactorizationRecommender):
    """ ALTERNATING LEAST SQUARE MATRIX FACTORIZATION RECOMMENDER SYSTEM ALGORITHM """

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True, alpha=29.4, lambda_val=4.98, latent_factors=400,
                 iterations=15): # orignial 460, 20

        super().__init__(URM, ICM, exclude_seen)

        self.alpha = alpha
        self.lambda_val = lambda_val
        self.latent_factors = latent_factors
        self.iterations = iterations
        self.Cui = self.URM.multiply(alpha).astype('double')

    def fit(self):

        """ Fits the ALS MF model """

        self._train()

    def _train(self, verbose=True):

        """ Trains the ALS MF model """

        start_time = time.time()

        if verbose:
            print('ALS training started...')

        user_size, item_size = self.Cui.shape
        self.user_factors = np.random.rand(user_size, self.latent_factors)
        self.item_factors = np.random.rand(item_size, self.latent_factors)

        Cui, Ciu = self.Cui.tocsr(), self.Cui.T.tocsr()

        for iteration in range(self.iterations):
            iter_start_time = time.time()

            least_squares_cg(Cui, self.user_factors, self.item_factors, self.lambda_val)
            least_squares_cg(Ciu, self.item_factors, self.user_factors, self.lambda_val)
            print('iteration {} of {} --> computed in {:.2f} minutes'.format(iteration + 1,
                                                                             self.iterations,
                                                                             (time.time() - iter_start_time) / 60))

        if verbose:
            print('ALS Matrix Factorization training computed in {:.2f} minutes'
                  .format((time.time() - start_time) / 60))

