import numpy as np
import scipy.sparse as sp

from recommenders.collaborativebasedfiltering import ItemBasedCFRecommender, UserBasedCFRecommender
from recommenders.contentbasedfiltering import CBFRecommender
from recommenders.mf_ials import ALSMFRecommender
from recommenders.recommender import Recommender
from recommenders.slimbpr import SLIM_BPR_Cython
from recommenders.sslimrmse import SSLIMRMSERecommender
from recommenders.svd import SVDRecommender


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

        self.IBCFweight = None
        self.UBCFweight = None
        self.LFMCFweight = None
        self.CBFweight = None
        self.SSLIMweight = None
        self.ALSweight = None
        self.SLIMBPRweight = None
        self.SVDweight = None

    def set_weights(self, IBCFweight=0.018, UBCFweight=1, LFMCFweight=0.0, CBFweight=1, SSLIMweight=0.0,
                    ALSweight=0.73, SLIMBPRweight=1, SVDweight=0.0):

        """ Sets the weights for every algorithm involved in the hybrid recommender """

        self.IBCFweight = IBCFweight
        self.UBCFweight = UBCFweight
        self.LFMCFweight = LFMCFweight
        self.CBFweight = CBFweight
        self.SSLIMweight = SSLIMweight
        self.ALSweight = ALSweight
        self.SLIMBPRweight = SLIMBPRweight
        self.SVDweight = SVDweight

    def fit(self):

        # if self.LFMCFweight is not None:
        #     ### LIGHTFM CF ###
        #     self.LFMCF = LightFMRecommender(self.URM_train, self.URM_test,
        #                                     self.URM_validation, self.target_playlists, subfolder=self.subfolder)
        #     self.LFMCF.fit()
        #     self.LFMCF._train()

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
        ]

        recommenders = [
            self.ItemBased,
            self.UserBased,
            self.ContentBased,
            self.SSLIM,
            self.ALS,
            self.SLIMBPR,
            self.SVD,
        ]
        predicted_ratings = np.zeros(shape=self.URM.shape[1], dtype=np.float32)

        for recommender, weight in zip(recommenders, weights):
            if weight > 0.0:
                ratings = recommender.compute_predicted_ratings(user_id)
                tops = get_tops(ratings, self.k)
                predicted_ratings[tops] += np.multiply(ratings[tops], weight)

        # if self.IBCFweight is not 0.0:
        #     itemcf_exp_ratings = self.ItemBased.compute_predicted_ratings(user_id)
        #     tops = get_tops(itemcf_exp_ratings, self.k)
        #     predicted_ratings[tops] += np.multiply(itemcf_exp_ratings[tops], self.IBCFweight)
        #
        # if self.UBCFweight is not 0.0:
        #     usercf_exp_ratings = self.UserBased.compute_predicted_ratings(user_id)
        #     tops = get_tops(usercf_exp_ratings, self.k)
        #     predicted_ratings[tops] += np.multiply(usercf_exp_ratings[tops], self.UBCFweight)
        #
        # # if self.LFMCFweight is not None:
        # #     lightfm_exp_ratings = self.LFMCF.compute_predicted_ratings(playlist_id=playlist_id)
        # #     tops = get_tops(lightfm_exp_ratings, self.k)
        # #     predicted_ratings[tops] += np.multiply(lightfm_exp_ratings[tops], self.LFMCFweight)
        #
        # if self.CBFweight is not 0.0:
        #     cbf_exp_ratings = self.ContentBased.compute_predicted_ratings(user_id)
        #     tops = get_tops(cbf_exp_ratings, self.k)
        #     predicted_ratings[tops] += np.multiply(cbf_exp_ratings[tops], self.CBFweight)
        #
        # if self.ALSweight is not 0.0:
        #     mfals_exp_ratings = self.ALS.compute_predicted_ratings(user_id)
        #     tops = get_tops(mfals_exp_ratings, self.k)
        #     predicted_ratings[tops] += np.multiply(mfals_exp_ratings[tops], self.ALSweight)
        #
        # if self.SSLIMweight is not 0.0:
        #     slimrmse_exp_ratings = self.SSLIM.compute_predicted_ratings(user_id)
        #     tops = get_tops(slimrmse_exp_ratings, self.k)
        #     predicted_ratings[tops] += np.multiply(slimrmse_exp_ratings[tops], self.SSLIMweight)
        #
        # if self.SLIMBPRweight is not 0.0:
        #     slimbpr_exp_ratings = self.SLIMBPR.compute_predicted_ratings(user_id)
        #     tops = get_tops(slimbpr_exp_ratings, self.k)
        #     predicted_ratings[tops] += np.multiply(slimbpr_exp_ratings[tops], self.SLIMBPRweight)
        #
        # if self.SVDweight is not 0.0:
        #     svd_exp_ratings = self.SVD.compute_predicted_ratings(user_id)
        #     tops = get_tops(svd_exp_ratings, self.k)
        #     predicted_ratings[tops] += np.multiply(svd_exp_ratings[tops], self.SVDweight)

        return predicted_ratings
