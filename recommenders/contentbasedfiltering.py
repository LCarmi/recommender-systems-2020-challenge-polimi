from recommenders.recommender import Recommender
from utils.official.Compute_Similarity import Compute_Similarity
import numpy as np
from utils.official.IR_feature_weighting import apply_feature_weighting


class CBFRecommender(Recommender):

    def __init__(self, URM, ICM, exclude_seen=True, topK=165, shrink=475, normalize=False, similarity="cosine",
                 asymmetric_alpha=0.67, feature_weighting="TF-IDF-Transpose", K=39.3, B=0.80):
        super().__init__(URM, ICM, exclude_seen)
        self.ICM = ICM
        self.W_sparse = None

        self.topK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = similarity
        self.asymmetric_alpha = asymmetric_alpha
        self.feature_weighting = feature_weighting
        self.K = K
        self.B = B

    def fit(self):
        self.ICM = apply_feature_weighting(self.ICM, self.feature_weighting, K=self.K, B=self.B)

        similarity_object = Compute_Similarity(self.ICM.T,
                                               shrink=self.shrink,
                                               topK=self.topK,
                                               normalize=self.normalize,
                                               similarity=self.similarity,
                                               asymmetric_alpha=self.asymmetric_alpha)

        self.W_sparse = similarity_object.compute_similarity()

        # Precompute URM
        self.predicted_URM = self.URM.dot(self.W_sparse)
