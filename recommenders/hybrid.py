import numpy as np
import scipy.sparse as sp

from recommenders.collaborativebasedfiltering import UserBasedCFRecommender, ItemBasedCFRecommender
from recommenders.contentbasedfiltering import CBFRecommender
from recommenders.mf_ials import ALSMFRecommender
from recommenders.sslimrmse import SSLIMRMSERecommender
from recommenders.svd import SVDRecommender
from recommenders.recommender import Recommender
from recommenders.slimbpr import SLIM_BPR_Cython
from recommenders.lightfm import LightFMRecommender
from recommenders.p3alpha import P3alphaRecommender
from recommenders.test import TopPopRecommender

def get_tops(ratings, k):
    """ Returns an array of k best tracks according to the ratings provided """

    # top k indices in sparse order
    ind = np.argpartition(ratings, -k)[-k:]
    f = np.flip(np.argsort(ratings[ind]))
    return ind[f]


class HybridRecommender(Recommender):
    """ HYBRID RECOMMENDER SYSTEM """

    N_CONFIG = 0

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True, k=None):

        super().__init__(URM, ICM, exclude_seen)
        self._set_k(k)
        self.normalize = None

        self.IBCFweight = None
        self.UBCFweight = None
        self.LFMCFweight = None
        self.CBFweight = None
        self.SSLIMweight = None
        self.ALSweight = None
        self.SLIMBPRweight = None
        self.SVDweight = None
        self.P3weight = None
        self.TopPopweight = 0.01

    def set_weights(self, IBCFweight=0.018, UBCFweight=1, LFMCFweight=0.0, CBFweight=1, SSLIMweight=0.0,
                    ALSweight=0.73, SLIMBPRweight=1, SVDweight=0.0, P3weight=1.0, normalize=False):

        """ Sets the weights for every algorithm involved in the hybrid recommender """

        self.IBCFweight = IBCFweight
        self.UBCFweight = UBCFweight
        self.LFMCFweight = LFMCFweight
        self.CBFweight = CBFweight
        self.SSLIMweight = SSLIMweight
        self.ALSweight = ALSweight
        self.SLIMBPRweight = SLIMBPRweight
        self.SVDweight = SVDweight
        self.P3weight = P3weight

        self.normalize = normalize


    def fit(self):

        ### TopPop ###
        self.TopPop = TopPopRecommender()
        self.TopPop.fit(self.URM)

        ### Item-based collaborative filtering ###
        self.ItemBased = ItemBasedCFRecommender(self.URM, self.ICM)
        self.ItemBased.fit()

        ### User-based collaborative filtering ###
        self.UserBased = UserBasedCFRecommender(self.URM, self.ICM)
        self.UserBased.fit()

        ### Content-based filtering ###
        self.ContentBased = CBFRecommender(self.URM, self.ICM)
        self.ContentBased.fit()

        ### ALS Matrix Factorization ###
        self.ALS = ALSMFRecommender(self.URM, self.ICM)
        self.ALS.fit()

        ### SLIM BPR SGD ###
        self.SLIMBPR = SLIM_BPR_Cython(self.URM, self.ICM)
        self.SLIMBPR.fit()

        ### SSLIM RMSE ###
        self.SSLIM = SSLIMRMSERecommender(self.URM, self.ICM)
        self.SSLIM.fit()

        ### SVD ###
        self.SVD = SVDRecommender(self.URM, self.ICM)
        self.SVD.fit()

        ### LIGHTFM CF ###
        self.LFMCF = LightFMRecommender(self.URM, self.ICM)
        self.LFMCF.fit()

        ### P3ALPHA CF ###
        self.P3 = P3alphaRecommender(self.URM, self.ICM)
        self.P3.fit()

        self.set_weights()

    def _set_k(self, k):

        """ Set the k value """

        if k is None:
            self.k = self.URM.shape[0]
        else:
            self.k = k

    def compute_predicted_ratings(self, user_id):

        """ Computes predicted ratings across all different recommender algorithms """

        weights = [
            self.IBCFweight,
            self.UBCFweight,
            self.CBFweight,
            self.SSLIMweight,
            self.ALSweight,
            self.SLIMBPRweight,
            self.SVDweight,
            self.LFMCFweight,
            self.P3weight,
            self.TopPopweight,
        ]

        recommenders = [
            self.ItemBased,
            self.UserBased,
            self.ContentBased,
            self.SSLIM,
            self.ALS,
            self.SLIMBPR,
            self.SVD,
            self.LFMCF,
            self.P3,
            self.TopPop
        ]
        predicted_ratings = np.zeros(shape=self.URM.shape[1], dtype=np.float32)

        for recommender, weight in zip(recommenders, weights):
            if weight > 0.0:
                ratings = recommender.compute_predicted_ratings(user_id)
                if self.normalize:
                    ratings *= 1.0 / ratings.max()
                tops = get_tops(ratings, self.k)
                predicted_ratings[tops] += np.multiply(ratings[tops], weight)

        return predicted_ratings
