import numpy as np
import scipy.sparse as sp
import time

from recommenders.collaborativebasedfiltering import UserBasedCFRecommender, ItemBasedCFRecommender
from recommenders.contentbasedfiltering import CBFRecommender
from recommenders.mf_ials import ALSMFRecommender, ImplicitALSRecommender
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
    return ind


class HybridRecommender(Recommender):
    """ HYBRID RECOMMENDER SYSTEM """

    N_CONFIG = 0

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True):

        super().__init__(URM, ICM, exclude_seen)
        self.normalize = None
        self.recommenders = {}
        self.weights = {}

    def fit(self, TopPopweight=0.001, IBCFweight=0.018, UBCFweight=1, LFMCFweight=0.0, CBFweight=1, SSLIMweight=0.0,
            ALSweight=0.73, SLIMBPRweight=0.0, SVDweight=0.0, P3weight=1.0, normalize=False):

        """ Sets the weights for every algorithm involved in the hybrid recommender """

        self.weights = {
            TopPopRecommender: TopPopweight,
            UserBasedCFRecommender: UBCFweight,
            ItemBasedCFRecommender: IBCFweight,
            CBFRecommender: CBFweight,
            LightFMRecommender: LFMCFweight,

            SSLIMRMSERecommender: SSLIMweight,
            SLIM_BPR_Cython: SLIMBPRweight,

            SVDRecommender: SVDweight,
            ImplicitALSRecommender: ALSweight,
            # ALSMFRecommender: ALSweight,

            P3alphaRecommender: P3weight
        }

        self.normalize = normalize

        for rec_class in self.weights.keys():
            if self.weights[rec_class] > 0.0:
                if rec_class not in self.recommenders:
                    start = time.time()
                    temp_rec = rec_class(self.URM, self.ICM)
                    temp_rec.fit()
                    self.recommenders[rec_class] = temp_rec
                    end = time.time()
                    print(
                        "Fitted new instance of {}. Employed time: {} seconds".format(rec_class.__name__, end - start))

    def compute_predicted_ratings(self, user_id):

        """ Computes predicted ratings across all different recommender algorithms """

        predicted_ratings = np.zeros(shape=self.URM.shape[1], dtype=np.float32)

        relevant_items = self.URM.indices[self.URM.indptr[user_id]:self.URM.indptr[user_id + 1]]
        if len(relevant_items) > 0:
            for rec_class in self.recommenders.keys():
                if self.weights[rec_class] > 0.0:
                    ratings = self.recommenders[rec_class].compute_predicted_ratings(user_id)
                    if self.normalize:
                        ratings *= 1.0 / ratings.max()
                    predicted_ratings += np.multiply(ratings, self.weights[rec_class])
        else:
            predicted_ratings = self.recommenders[TopPopRecommender].compute_predicted_ratings(user_id)

        return predicted_ratings


class HybridRecommenderWithTopK(Recommender):
    """ HYBRID RECOMMENDER SYSTEM """

    N_CONFIG = 0

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True, k=5000):

        super().__init__(URM, ICM, exclude_seen)
        self.k = k
        self.normalize = None
        self.weights = {}
        self.precomputed_ratings = {}
        self.precomputed_ratings_item_masks = {}

    def fit(self, TopPopweight=0.001, IBCFweight=0.018, UBCFweight=1, LFMCFweight=0.0, CBFweight=1, SSLIMweight=0.0,
            ALSweight=0.73, SLIMBPRweight=0.0, SVDweight=0.0, P3weight=1.0, normalize=True):

        """ Sets the weights for every algorithm involved in the hybrid recommender """

        self.weights = {
            TopPopRecommender: TopPopweight,
            UserBasedCFRecommender: UBCFweight,
            ItemBasedCFRecommender: IBCFweight,
            CBFRecommender: CBFweight,
            LightFMRecommender: LFMCFweight,

            SSLIMRMSERecommender: SSLIMweight,
            SLIM_BPR_Cython: SLIMBPRweight,

            SVDRecommender: SVDweight,
            ImplicitALSRecommender: ALSweight,

            P3alphaRecommender: P3weight
        }

        self.normalize = normalize

        for rec_class in self.weights.keys():
            if self.weights[rec_class] > 0.0 or rec_class == TopPopRecommender:
                if rec_class not in self.precomputed_ratings:
                    start = time.time()

                    temp_rec = rec_class(self.URM, self.ICM)
                    temp_rec.fit()

                    self.precomputed_ratings[rec_class] = np.zeros((self.URM.shape[0], self.k), np.float32)
                    self.precomputed_ratings_item_masks[rec_class] = np.zeros((self.URM.shape[0], self.k), np.int)
                    for user_id in np.arange(self.URM.shape[0]):
                        ratings, mask = temp_rec.compute_predicted_ratings_top_k(user_id, self.k)
                        self.precomputed_ratings[rec_class][user_id] = ratings
                        self.precomputed_ratings_item_masks[rec_class][user_id] = mask
                    end = time.time()
                    print(
                        "Fitted new instance of {}. Employed time: {} seconds".format(rec_class.__name__, end - start))

    def compute_predicted_ratings(self, user_id):

        def soft_max(z):
            t = np.exp(z)
            a = np.exp(z) / np.sum(t)
            return a
        """ Computes predicted ratings across all different recommender algorithms """

        predicted_ratings = np.zeros(shape=self.URM.shape[1], dtype=np.float32)

        relevant_items = self.URM.indices[self.URM.indptr[user_id]:self.URM.indptr[user_id + 1]]

        for rec_class in self.precomputed_ratings.keys():
            if self.weights[rec_class] > 0.0:
                if self.normalize:
                    denominator = self.precomputed_ratings[rec_class][user_id].max() - self.precomputed_ratings[rec_class][user_id].min()
                    if denominator not in [np.nan, np.inf, 0, 0.0]:
                        predicted_ratings[self.precomputed_ratings_item_masks[rec_class][user_id]] += np.multiply(
                            self.precomputed_ratings[rec_class][user_id], self.weights[rec_class]) / denominator
                    #else:
                    #    print("User {} got an empty prediction by {} even if his relevant items are {}. Denominator is {}".format(user_id, rec_class, relevant_items, denominator))
                else:
                    predicted_ratings[self.precomputed_ratings_item_masks[rec_class][user_id]] += np.multiply(
                        self.precomputed_ratings[rec_class][user_id], self.weights[rec_class])

        return predicted_ratings
