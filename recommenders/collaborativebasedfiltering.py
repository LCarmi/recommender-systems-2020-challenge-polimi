from recommenders.recommender import Recommender
from utils.official.Compute_Similarity import Compute_Similarity
from utils.official.IR_feature_weighting import okapi_BM_25, TF_IDF
from utils.official.Recommender_utils import check_matrix

import numpy as np


class UserBasedCFRecommender(Recommender):

    def __init__(self, URM, ICM, exclude_seen=True, topK=350, shrink=8.9, normalize=True, similarity="tanimoto",
                 asymmetric_alpha=0.77, feature_weighting=None):
        super().__init__(URM, ICM, exclude_seen)
        self.W_sparse = None

        self.topK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = similarity
        self.asymmetric_alpha = asymmetric_alpha
        self.feature_weighting = feature_weighting

    def fit(self):

        if self.feature_weighting == "BM25":
            self.URM = self.URM.astype(np.float32)
            self.URM = okapi_BM_25(self.URM.T).T
            self.URM = check_matrix(self.URM, 'csr')

        elif self.feature_weighting == "TF-IDF":
            self.URM = self.URM.astype(np.float32)
            self.URM = TF_IDF(self.URM.T).T
            self.URM = check_matrix(self.URM, 'csr')

        similarity_object = Compute_Similarity(self.URM.T,
                                               shrink=self.shrink,
                                               topK=self.topK,
                                               normalize=self.normalize,
                                               similarity=self.similarity,
                                               asymmetric_alpha=self.asymmetric_alpha)

        self.W_sparse = similarity_object.compute_similarity()

        # Precompute URM
        self.predicted_URM = self.W_sparse.dot(self.URM)


class ItemBasedCFRecommender(Recommender):

    def __init__(self, URM, ICM, exclude_seen=True, topK=3333, shrink=500, normalize=True, similarity="cosine",
                 asymmetric_alpha=0.5, feature_weighting='TF-IDF'):
        super().__init__(URM, ICM, exclude_seen)
        self.W_sparse = None

        self.topK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = similarity
        self.asymmetric_alpha = asymmetric_alpha
        self.feature_weighting = feature_weighting

    def fit(self):

        if self.feature_weighting == "BM25":
            self.URM = self.URM.astype(np.float32)
            self.URM = okapi_BM_25(self.URM.T).T
            self.URM = check_matrix(self.URM, 'csr')

        elif self.feature_weighting == "TF-IDF":
            self.URM = self.URM.astype(np.float32)
            self.URM = TF_IDF(self.URM.T).T
            self.URM = check_matrix(self.URM, 'csr')

        similarity_object = Compute_Similarity(self.URM,
                                               shrink=self.shrink,
                                               topK=self.topK,
                                               normalize=self.normalize,
                                               similarity=self.similarity,
                                               asymmetric_alpha=self.asymmetric_alpha)

        self.W_sparse = similarity_object.compute_similarity()

        # Precompute URM
        self.predicted_URM = self.URM.dot(self.W_sparse)
