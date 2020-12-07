import numpy as np

from recommenders.recommender import Recommender
import scipy.sparse as sp


class RandomRecommender():
    def __init__(self):
        self.n_items = None

    def fit(self, URM_train):
        self.n_items = URM_train.shape[1]

    def recommend(self, user_id, at=5):
        recommended_items = np.random.choice(self.n_items, at)

        return recommended_items


class TopPopRecommender(Recommender):
    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True):
        super().__init__(URM, ICM, exclude_seen)
        self.popular_items = None
        self.item_popularity = None

    def fit(self):
        self.item_popularity = np.ediff1d(self.URM.copy().tocsc().indptr)

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popular_items = np.argsort(self.item_popularity)
        self.popular_items = np.flip(self.popular_items, axis=0)

    def compute_predicted_ratings(self, user_id):
        return self.item_popularity.astype(np.float)