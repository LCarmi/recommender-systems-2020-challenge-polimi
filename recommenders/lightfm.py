import time
import numpy as np

from recommenders.recommender import Recommender
from lightfm import LightFM
import scipy.sparse as sp
from utils.official.IR_feature_weighting import apply_feature_weighting
from utils.official.Recommender_utils import check_matrix


class LightFMRecommender(Recommender):
    N_CONFIG = 0

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True,
                 item_alpha=1e-4, user_alpha=1e-6, learning_schedule='adadelta', loss='warp', feature_weighting="TF-IDF",
                 num_components=280, epochs=30, threads=1, K=1.2, B=0.75):

        super().__init__(URM, ICM, exclude_seen)

        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.learning_schedule = learning_schedule
        self.num_components = num_components
        self.epochs = epochs
        self.threads = threads
        self.loss = loss
        self.feature_weighting = feature_weighting
        self.K = K
        self.B = B

    def fit(self):
        self.URM = apply_feature_weighting(self.URM, self.feature_weighting, K=self.K, B=self.B)

        self._train()

    def _train(self, verbose=True):
        start_time = time.time()

        if verbose:
            print("LightFM training started!")

        # Let's fit a WARP model: these generally have the best performance.
        self.model = LightFM(loss=self.loss,
                             item_alpha=self.item_alpha,
                             user_alpha=self.user_alpha,
                             learning_schedule=self.learning_schedule,
                             no_components=self.num_components)

        # Run 3 epochs and time it.
        self.model = self.model.fit(self.URM, epochs=self.epochs, num_threads=self.threads)
        if verbose:
            print("LightFM training model fitted in {:.2f} seconds".format(time.time() - start_time))

    def compute_predicted_ratings(self, user_id):
        return self.model.predict(user_ids=int(user_id), item_ids=np.arange(self.URM.shape[1]), item_features=None,
                                  user_features=None, num_threads=self.threads)
