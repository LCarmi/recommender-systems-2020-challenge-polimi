import numpy as np
import scipy.sparse as sp

from recommenders.test import TopPopRecommender


class Recommender:

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True):
        if not sp.isspmatrix_csr(URM):
            raise TypeError(f"We expected a CSR matrix, we got {type(URM)}")
        self.URM = URM
        self.ICM = ICM
        self.predicted_URM = None
        self.exclude_seen = exclude_seen
        self.recommendations = None

    def fit(self):
        """
        Performs fitting and training of the recommender
        Prepares the predicted_URM matrix
        All needed parameters must be passed through init
        :return: Nothing
        """
        raise NotImplementedError()

    def recommend(self, user_id, at=10):
        """
        Provides a list of 'at' recommended items for the given user
        :param user_id: id for which provide recommendation
        :param at: how many items have to be recommended
        :return: recommended items list
        """

        predicted_ratings = self.compute_predicted_ratings(user_id)

        if self.exclude_seen:
            predicted_ratings = self.__filter_seen(user_id, predicted_ratings)

        # ideally do
        # ordered_items = np.flip(np.argsort(predicted_ratings))
        # recommended_items = ordered_items[:at]
        # return recommended_items

        # BUT O(NlogN) -> MORE EFFICIENT O(N+KlogK)

        # top k indices in sparse order
        ind = np.argpartition(predicted_ratings, -at)[-at:]
        # support needed to correctly index
        f = np.flip(np.argsort(predicted_ratings[ind]))
        # assert((predicted_ratings[recommended_items] == predicted_ratings[ind[f]]).all())
        return ind[f]

    def compute_predicted_ratings(self, user_id):

        """ Compute the predicted ratings for a given user_id """

        return self.predicted_URM[user_id].toarray().ravel()

    def __filter_seen(self, user_id, predicted_ratings):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        predicted_ratings[user_profile] = -np.inf

        return predicted_ratings


class MatrixFactorizationRecommender(Recommender):
    """ ABSTRACT MATRIX FACTORIZATION RECOMMENDER """

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True):
        super().__init__(URM, ICM, exclude_seen)
        self.user_factors = None  # playlist x latent_factors
        self.item_factors = None  # tracks x latent_factors

    def compute_predicted_ratings(self, user_id):
        """ Compute predicted ratings for a given playlist in case of
        matrix factorization algorithm """

        return np.dot(self.user_factors[user_id], self.item_factors.T)

    def fit(self):
        raise NotImplementedError()
