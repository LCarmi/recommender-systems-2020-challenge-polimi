import numpy as np
import scipy.sparse as sp
import time

from recommenders.collaborativebasedfiltering import *
from recommenders.contentbasedfiltering import *
from recommenders.mf_ials import *
from recommenders.sslimrmse import *
from recommenders.svd import *
from recommenders.test import *
from recommenders.slimbpr import *
from recommenders.lightfm import *
from recommenders.graphbased import *


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

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True, k=300):  # K=2000 in K_cross long

        super().__init__(URM, ICM, exclude_seen)
        self.k = k
        self.normalize = None
        self.weights = {}
        self.weight_vec = None
        self.ratings = None
        self.rating_masks = None

        self.low_data_rec_idxs = None
        self.threshold = 3

    def fit(self, TopPopweight=0.29329, IBCFweight=0.108121, UBCFweight=0.0230483, IBCFweightSI=0.0, UBCFweightSI=0.26704, LFMCFweight=0.0,
            CBFweight=0.209204, SSLIMweight=0.0, ALSweight=4.83469, SLIMBPRweight=0.0, SVDweight=0.0, P3weight=0.0,
            RP3weight=0.875458, SVDweightSI=0.0, P3weightSI=0.0, RP3weightSI=2.87098, normalize=True, threshold=0):

        """ Sets the weights for every algorithm involved in the hybrid recommender """

        self.weights = {
            TopPopRecommender: TopPopweight,
            UserBasedCFRecommender: UBCFweight,
            ItemBasedCFRecommender: IBCFweight,
            UserBasedCFRecommenderSI: UBCFweightSI,
            #ItemBasedCFRecommenderSI: IBCFweightSI,
            CBFRecommender: CBFweight,

            # LightFMRecommender: LFMCFweight,

            #SSLIMRMSERecommender: SSLIMweight,
            #SLIM_BPR_Cython: SLIMBPRweight,

            #SVDRecommender: SVDweight,
            #SVDRecommenderSI: SVDweightSI,

            ImplicitALSRecommender: ALSweight,

            #P3alphaRecommender: P3weight,
            RP3betaRecommender: RP3weight,
            #P3alphaRecommenderSI: P3weightSI,
            RP3betaRecommenderSI: RP3weightSI
        }
        # Cross check not to assign positive weight to ignored recommenders
        assert (LFMCFweight == 0)
        assert (SVDweight == 0)
        assert (SSLIMweight == 0)

        self.threshold = threshold
        self.normalize = normalize

        l = len(self.weights.keys())
        self.weight_vec = np.zeros(len(self.weights.keys()))

        if self.ratings is None:  # first time fitted -> compute predicted ratings
            self.ratings = np.zeros((l, self.URM.shape[0], self.k), np.float32)
            self.rating_masks = np.zeros(((l, self.URM.shape[0], self.k)), dtype=np.int)

            for i, rec_class in enumerate(self.weights.keys()):
                start = time.time()

                temp_rec = rec_class(self.URM, self.ICM, exclude_seen=False)
                temp_rec.fit()

                self.weight_vec[i] = self.weights[rec_class]
                for user_id in np.arange(self.URM.shape[0]):
                    ratings, mask = temp_rec.compute_predicted_ratings_top_k(user_id, self.k)
                    self.ratings[i, user_id, :] = ratings
                    self.rating_masks[i, user_id, :] = mask
                end = time.time()
                print(
                    "Fitted new instance of {}. Employed time: {} seconds".format(rec_class.__name__, end - start))
            item_recs = [
                CBFRecommender, RP3betaRecommender, TopPopRecommender
            ]
            low_data_rec_idxs = []
            for j, rec_class in enumerate(self.weights.keys()):
                if rec_class in item_recs:
                    low_data_rec_idxs.append(j)
            self.low_data_rec_idxs = np.array(low_data_rec_idxs)

        else:  # second or more time fitted -> only change weights, reuse same predicted ratings :)
            for i, rec_class in enumerate(self.weights.keys()):
                self.weight_vec[i] = self.weights[rec_class]

    def compute_predicted_ratings(self, user_id):

        """ Computes predicted ratings across all different recommender algorithms """
        # Allocate space for results
        predicted_ratings = np.zeros(shape=self.URM.shape[1], dtype=np.float32)

        # Select correctly the recommenders to use for the recommendation depending on profile lenght
        user_seen_items = self.URM.indices[self.URM.indptr[user_id]:self.URM.indptr[user_id + 1]]
        if len(user_seen_items) >= self.threshold:
            indices = np.arange(self.ratings.shape[0])
        else:
            indices = self.low_data_rec_idxs

        # Slice the precomputed ratings tensor to retrieve the prediction for the user_id
        for i in indices:  # for each recommender
            # Apply L1 Normalization
            denominator = 0
            if self.normalize:
                denominator = np.linalg.norm(self.ratings[i, user_id, :])
            # Avoid strange cases (especially when user has only 1 interaction and it has gone in the test set)
            if denominator in [np.nan, np.inf, 0, 0.0]:
                denominator = 1
            predicted_ratings[self.rating_masks[i, user_id, :]] += self.ratings[i, user_id, :] * (
                    self.weight_vec[i] / denominator)

        return predicted_ratings

    # def _sample_triplet(self):
    #
    #     non_empty_user = False
    #
    #     while not non_empty_user:
    #         user_id = np.random.choice(self.URM.shape[0])
    #         user_seen_items = self.URM.indices[self.URM.indptr[user_id]:self.URM.indptr[user_id + 1]]
    #
    #         if len(user_seen_items) > 0:
    #             non_empty_user = True
    #
    #     pos_item_id = np.random.choice(user_seen_items)
    #
    #     neg_item_selected = False
    #
    #     # It's faster to just try again then to build a mapping of the non-seen items
    #     while not neg_item_selected:
    #         neg_item_id = np.random.randint(0, self.URM.shape[1])
    #
    #         if neg_item_id not in user_seen_items:
    #             neg_item_selected = True
    #
    #     return user_id, pos_item_id, neg_item_id
    #
    # def train(self, n_steps):
    #     learning_rate = 1e-3
    #     for t in range(n_steps):
    #         print("Train epoch: starting iteration {}".format(t))
    #         # Sample triplet
    #         user_id, pos_item_id, neg_item_id = self._sample_triplet()
    #
    #         terms = np.zeros(len(self.precomputed_ratings.keys()))
    #
    #         for i in self.ratings.shape[0]:
    #             # Prediction
    #             try:
    #                 item_id = np.argwhere(self.rating_masks[i, user_id, :] == pos_item_id).flatten()[0]
    #                 x_ui = self.ratings[i, user_id, item_id]
    #             except IndexError:
    #                 x_ui = 0
    #             try:
    #                 item_id = np.argwhere(self.rating_masks[i, user_id, :] == neg_item_id).flatten()[0]
    #                 x_uj = self.ratings[i, user_id, item_id]
    #             except IndexError:
    #                 x_uj = 0
    #
    #             # Gradient
    #             terms[i] = x_ui - x_uj
    #
    #         terms *= (1 - 1 / (1 + np.exp(-terms.sum())))
    #         self.weight_vec += learning_rate * terms


class SequentialHybrid(Recommender):

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True):
        super().__init__(URM, ICM, exclude_seen)

    def fit(self):
        r1 = UserBasedCFRecommender(self.URM, self.ICM, False)
        r1.fit()
        r2 = RP3betaRecommender(r1.predicted_URM, self.ICM, False)
        r2.fit()
        self.predicted_URM = r2.predicted_URM


class EnrichingHybrid(Recommender):

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True, k=3):
        super().__init__(URM.copy(), ICM, exclude_seen)
        self.k = k

    def fit(self):
        r1 = UserBasedCFRecommender(self.URM, self.ICM, True)
        r1.fit()
        extra_rows = []
        extra_cols = []
        extra_data = []
        for user_id in range(self.URM.shape[0]):
            extra_cols.extend(r1.recommend(user_id, at=self.k))
            extra_rows.extend([user_id for _ in range(self.k)])
            extra_data.extend([1 for _ in range(self.k)])

        URM_coo = self.URM.tocoo()
        URM_csr = sp.csr_matrix((np.append(URM_coo.data, extra_data),
                                 (np.append(URM_coo.row, extra_rows), np.append(URM_coo.col, extra_cols))),
                                shape=URM_coo.shape)

        r2 = RP3betaRecommender(URM_csr, self.ICM, False)
        r2.fit()

        self.predicted_URM = r2.predicted_URM
