from recommenders.recommender import Recommender
from utils.official.Compute_Similarity import Compute_Similarity
from utils.official.IR_feature_weighting import apply_feature_weighting


class UserBasedCFRecommender(Recommender):

    def __init__(self, URM, ICM, exclude_seen=True, topK=242, shrink=9.02, normalize=True, similarity="tanimoto",
                 asymmetric_alpha=0.77, feature_weighting="TF-IDF", K=40, B=0.75):
        super().__init__(URM, ICM, exclude_seen)
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

        self.URM = apply_feature_weighting(self.URM, self.feature_weighting, K=self.K, B=self.B)

        similarity_object = Compute_Similarity(self.URM.T,
                                               shrink=self.shrink,
                                               topK=self.topK,
                                               normalize=self.normalize,
                                               similarity=self.similarity,
                                               asymmetric_alpha=self.asymmetric_alpha)

        self.W_sparse = similarity_object.compute_similarity()

        # Precompute URM
        self.predicted_URM = self.W_sparse.dot(self.URM)


class UserBasedCFRecommenderSI(UserBasedCFRecommender):
    def __init__(self, URM, ICM, exclude_seen=True, topK=165, shrink=9.33, normalize=True, similarity="tanimoto",
                 asymmetric_alpha=0.77, feature_weighting="TF-IDF", K=40, B=0.75, omega=0.9):

        super().__init__(URM, ICM, exclude_seen, topK, shrink, normalize, similarity, asymmetric_alpha, feature_weighting, K, B)
        self.add_side_information(omega)


class ItemBasedCFRecommender(Recommender):

    def __init__(self, URM, ICM, exclude_seen=True, topK=2900, shrink=860, normalize=True, similarity="cosine",
                 asymmetric_alpha=0.5, feature_weighting="TF-IDF-Transpose", K=1.2, B=0.75):
        super().__init__(URM, ICM, exclude_seen)
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
        self.URM = apply_feature_weighting(self.URM, self.feature_weighting, K=self.K, B=self.B)

        similarity_object = Compute_Similarity(self.URM,
                                               shrink=self.shrink,
                                               topK=self.topK,
                                               normalize=self.normalize,
                                               similarity=self.similarity,
                                               asymmetric_alpha=self.asymmetric_alpha)

        self.W_sparse = similarity_object.compute_similarity()

        # Precompute URM
        self.predicted_URM = self.URM.dot(self.W_sparse)


class ItemBasedCFRecommenderSI(ItemBasedCFRecommender):

    def __init__(self, URM, ICM, exclude_seen=True, topK=122, shrink=262, normalize=True, similarity="dice",
                 asymmetric_alpha=0.5, feature_weighting="TF-IDF", K=1.2, B=0.75, omega=20.39):
        super().__init__(URM, ICM, exclude_seen, topK, shrink, normalize, similarity, asymmetric_alpha, feature_weighting, K, B)

        self.add_side_information(omega)
