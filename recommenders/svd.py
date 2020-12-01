import time

import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd

from recommenders.recommender import MatrixFactorizationRecommender


class SVDRecommender(MatrixFactorizationRecommender):

    """ SINGULAR VALUE DECOMPOSITION MATRIX FACTORIZATION RECOMMENDER SYSTEM ALGORITHM """

    N_CONFIG = 0

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True, latent_factors=20, scipy=True):

        super().__init__(URM, ICM, exclude_seen)

        self.latent_factors = latent_factors
        self.scipy = scipy

    def fit(self):
        self._train()

    def _train(self, verbose=True):

        """ Trains the ALS MF model """

        start_time = time.time()

        if verbose:
            print('SVD training started...')

        if self.scipy:
            print('computing u, s, v  using scipy model ...')
            u, s, v = svds(self.URM.astype('float'), k=self.latent_factors, which='LM')
        else:
            print('computing u, s, v using sklearn model ...')
            u, s, v = randomized_svd(self.URM, n_components=self.latent_factors, random_state=None,
                                     power_iteration_normalizer='QR', n_iter=100)

        print('computing SVD expected urm ...')

        s = sp.diags(s)
        self.user_factors = u
        self.item_factors = s.dot(v).T

        if verbose:
            print('SVD Matrix Factorization training computed in {:.2f} minutes'
                  .format((time.time() - start_time) / 60))
