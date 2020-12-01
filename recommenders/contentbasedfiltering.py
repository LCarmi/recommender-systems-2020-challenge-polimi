from recommenders.recommender import Recommender
from utils.official.Compute_Similarity import Compute_Similarity
import numpy as np
from utils.official.IR_feature_weighting import okapi_BM_25, TF_IDF


class CBFRecommender(Recommender):

    def __init__(self, URM, ICM, exclude_seen=True, topK=121, shrink=1.0, normalize=True, similarity="jaccard",
                 asymmetric_alpha=0.67, feature_weighting="BM-25"):
        super().__init__(URM, ICM, exclude_seen)
        self.ICM = ICM
        self.W_sparse = None

        self.topK = topK
        self.shrink = shrink
        self.normalize = normalize
        self.similarity = similarity
        self.asymmetric_alpha = asymmetric_alpha
        self.feature_weighting = feature_weighting

    def fit(self):
        if self.feature_weighting == "BM25":
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = okapi_BM_25(self.ICM)

        elif self.feature_weighting == "TF-IDF":
            self.ICM = self.ICM.astype(np.float32)
            self.ICM = TF_IDF(self.ICM)

        similarity_object = Compute_Similarity(self.ICM.T,
                                               shrink=self.shrink,
                                               topK=self.topK,
                                               normalize=self.normalize,
                                               similarity=self.similarity,
                                               asymmetric_alpha=self.asymmetric_alpha)

        self.W_sparse = similarity_object.compute_similarity()

        # Precompute URM
        self.predicted_URM = self.URM.dot(self.W_sparse)
