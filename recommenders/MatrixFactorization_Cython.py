from recommenders.recommender import Recommender, MatrixFactorizationRecommender
from utils.official.Recommender_utils import check_matrix
from recommenders.Cython.MatrixFactorization_Cython_Epoch import MatrixFactorization_Cython_Epoch
import numpy as np
import scipy.sparse as sp
import sys

class _MatrixFactorization_Cython(MatrixFactorizationRecommender):

    def __init__(self, URM: sp.csr_matrix, ICM, algorithm_name="MF_BPR", exclude_seen=True,
                 epochs=300, batch_size=1000,
                 num_factors=400, positive_threshold_BPR=None,
                 learning_rate=0.001, use_bias=True,
                 sgd_mode='sgd',
                 negative_interactions_quota=0.0,
                 init_mean=0.0, init_std_dev=0.1,
                 user_reg=0.0, item_reg=0.0, bias_reg=0.0, positive_reg=0.0, negative_reg=0.0,
                 verbose=True
        ):

        super().__init__(URM, ICM, exclude_seen)
        self.n_users, self.n_items = self.URM.shape
        self.normalize = False
        self.algorithm_name = algorithm_name

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_factors = num_factors
        self.positive_threshold_BPR = positive_threshold_BPR
        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.sgd_mode = sgd_mode

        assert negative_interactions_quota >= 0.0 and negative_interactions_quota < 1.0, "{}: negative_interactions_quota must be a float value >=0 and < 1.0, provided was '{}'".format(
            self.RECOMMENDER_NAME, negative_interactions_quota)
        self.negative_interactions_quota = negative_interactions_quota

        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.bias_reg = bias_reg
        self.positive_reg = positive_reg
        self.negative_reg = negative_reg

        self.verbose = verbose

    def fit(self):

        if self.algorithm_name in ["FUNK_SVD", "ASY_SVD"]:

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(self.URM,
                                                                algorithm_name=self.algorithm_name,
                                                                n_factors=self.num_factors,
                                                                learning_rate=self.learning_rate,
                                                                sgd_mode=self.sgd_mode,
                                                                user_reg=self.user_reg,
                                                                item_reg=self.item_reg,
                                                                bias_reg=self.bias_reg,
                                                                batch_size=self.batch_size,
                                                                use_bias=self.use_bias,
                                                                init_mean=self.init_mean,
                                                                negative_interactions_quota=self.negative_interactions_quota,
                                                                init_std_dev=self.init_std_dev,
                                                                verbose=self.verbose)

        elif self.algorithm_name == "MF_BPR":

            # Select only positive interactions
            URM_train_positive = self.URM.copy()

            if self.positive_threshold_BPR is not None:
                URM_train_positive.data = URM_train_positive.data >= self.positive_threshold_BPR
                URM_train_positive.eliminate_zeros()

                assert URM_train_positive.nnz > 0, "MatrixFactorization_Cython: URM_train_positive is empty, positive threshold is too high"

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(URM_train_positive,
                                                                algorithm_name=self.algorithm_name,
                                                                n_factors=self.num_factors,
                                                                learning_rate=self.learning_rate,
                                                                sgd_mode=self.sgd_mode,
                                                                user_reg=self.user_reg,
                                                                positive_reg=self.positive_reg,
                                                                negative_reg=self.negative_reg,
                                                                batch_size=self.batch_size,
                                                                use_bias=self.use_bias,
                                                                init_mean=self.init_mean,
                                                                init_std_dev=self.init_std_dev,
                                                                verbose=self.verbose)
        self._prepare_model_for_validation()
        #self._update_best_model()

        epochs_current = 0
        lower_epochs = 0
        while epochs_current < self.epochs:
            # run an epoch
            self.cythonEpoch.epochIteration_Cython()
            epochs_current+=1

        self._prepare_model_for_validation()

        #self.user_factors = self.USER_factors_best
        #self.item_factors = self.ITEM_factors_best

        #if self.use_bias:
        #    self.USER_bias = self.USER_bias_best
        #    self.ITEM_bias = self.ITEM_bias_best
        #    self.GLOBAL_bias = self.GLOBAL_bias_best

        sys.stdout.flush()

    def _prepare_model_for_validation(self):
        self.user_factors = self.cythonEpoch.get_USER_factors()
        self.item_factors = self.cythonEpoch.get_ITEM_factors()

        if self.use_bias:
            self.USER_bias = self.cythonEpoch.get_USER_bias()
            self.ITEM_bias = self.cythonEpoch.get_ITEM_bias()
            self.GLOBAL_bias = self.cythonEpoch.get_GLOBAL_bias()

    def _update_best_model(self):
        self.USER_factors_best = self.user_factors.copy()
        self.ITEM_factors_best = self.item_factors.copy()

        if self.use_bias:
            self.USER_bias_best = self.USER_bias.copy()
            self.ITEM_bias_best = self.ITEM_bias.copy()
            self.GLOBAL_bias_best = self.GLOBAL_bias

    def _run_epoch(self, num_epoch):
        self.cythonEpoch.epochIteration_Cython()



class MatrixFactorization_BPR_Cython(_MatrixFactorization_Cython):
    """
    Subclas allowing only for MF BPR
    """

    RECOMMENDER_NAME = "MatrixFactorization_BPR_Cython_Recommender"

    def __init__(self, *pos_args, **key_args):
        key_args["use_bias"] = False
        key_args["negative_interactions_quota"] = 0.0
        super(MatrixFactorization_BPR_Cython, self).__init__(*pos_args, algorithm_name="MF_BPR", **key_args)

    def fit(self, **key_args):
        super(MatrixFactorization_BPR_Cython, self).fit(**key_args)


class MatrixFactorization_FunkSVD_Cython(_MatrixFactorization_Cython):
    """
    Subclas allowing only for FunkSVD model

    Reference: http://sifter.org/~simon/journal/20061211.html

    Factorizes the rating matrix R into the dot product of two matrices U and V of latent factors.
    U represent the user latent factors, V the item latent factors.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin} \limits_{U,V}\frac{1}{2}||R - UV^T||^2_2 + \frac{\lambda}{2}(||U||^2_F + ||V||^2_F)
    Latent factors are initialized from a Normal distribution with given mean and std.

    """

    RECOMMENDER_NAME = "MatrixFactorization_FunkSVD_Cython_Recommender"

    def __init__(self, *pos_args, **key_args):
        super(MatrixFactorization_FunkSVD_Cython, self).__init__(*pos_args, algorithm_name="FUNK_SVD", **key_args)

    def fit(self, **key_args):
        super(MatrixFactorization_FunkSVD_Cython, self).fit(**key_args)


class MatrixFactorization_AsySVD_Cython(_MatrixFactorization_Cython):
    """
    Subclas allowing only for AsymmetricSVD model

    Reference: Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model (Koren, 2008)

    Factorizes the rating matrix R into two matrices X and Y of latent factors, which both represent item latent features.
    Users are represented by aggregating the latent features in Y of items they have already rated.
    Rating prediction is performed by computing the dot product of this accumulated user profile with the target item's
    latent factor in X.

    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j \in R}(r_{ij} - x_j^T \sum_{l \in R(i)} r_{il}y_l)^2 + \frac{\lambda}{2}(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})
    """

    RECOMMENDER_NAME = "MatrixFactorization_AsySVD_Cython_Recommender"

    def __init__(self, *pos_args, **key_args):

        if "batch_size" in key_args and key_args["batch_size"] > 1:
            print("{}: batch_size not supported for this recommender, setting to default value 1.".format(
                self.RECOMMENDER_NAME))

        key_args["batch_size"] = 1

        super(MatrixFactorization_AsySVD_Cython, self).__init__(*pos_args, algorithm_name="ASY_SVD", **key_args)

    def fit(self, **key_args):

        super(MatrixFactorization_AsySVD_Cython, self).fit(**key_args)

    def _prepare_model_for_validation(self):
        """
        AsymmetricSVD Computes two |n_items| x |n_features| matrices of latent factors
        ITEM_factors_Y must be used to estimate user's latent factors via the items they interacted with

        :return:
        """

        self.ITEM_factors_Y = self.cythonEpoch.get_USER_factors()
        self.user_factors = self._estimate_user_factors(self.ITEM_factors_Y)

        self.item_factors = self.cythonEpoch.get_ITEM_factors()

        if self.use_bias:
            self.USER_bias = self.cythonEpoch.get_USER_bias()
            self.ITEM_bias = self.cythonEpoch.get_ITEM_bias()
            self.GLOBAL_bias = self.cythonEpoch.get_GLOBAL_bias()

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self.ITEM_factors_Y_best = self.ITEM_factors_Y.copy()

        if self.use_bias:
            self.USER_bias_best = self.USER_bias.copy()
            self.ITEM_bias_best = self.ITEM_bias.copy()
            self.GLOBAL_bias_best = self.GLOBAL_bias

    def _estimate_user_factors(self, ITEM_factors_Y):

        profile_length = np.ediff1d(self.URM.indptr)
        profile_length_sqrt = np.sqrt(profile_length)

        # Estimating the USER_factors using ITEM_factors_Y
        if self.verbose:
            print("{}: Estimating user factors... ".format(self.algorithm_name))

        USER_factors = self.URM.dot(ITEM_factors_Y)

        # Divide every row for the sqrt of the profile length
        for user_index in range(self.n_users):

            if profile_length_sqrt[user_index] > 0:
                USER_factors[user_index, :] /= profile_length_sqrt[user_index]

        if self.verbose:
            print("{}: Estimating user factors... done!".format(self.algorithm_name))

        return USER_factors
