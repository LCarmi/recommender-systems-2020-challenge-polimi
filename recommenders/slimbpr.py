import sys
from recommenders.Cython.SLIM_BPR_Cython_Epoch import SLIM_BPR_Cython_Epoch
import utils
from recommenders.recommender import Recommender
from utils.official.IR_feature_weighting import apply_feature_weighting
from utils.official.Recommender_utils import check_matrix
from utils.official.Recommender_utils import similarityMatrixTopK


class SLIM_BPR_Cython(Recommender):

    def __init__(self, URM, ICM, exclude_seen=True,
                 epochs=150, symmetric=False,
                 batch_size=1000, lambda_i=0.872, lambda_j=0.364, learning_rate=1e-3, topK=295,
                 sgd_mode='adagrad', gamma=0.995, beta_1=0.9, beta_2=0.999,
                 patience=None, validation_split=0.2,
                 verbose=True,
                 omega=22, add_side_info=True, feature_weighting="TF-IDF",
                 ):
        if patience != None:
            URM_train, URM_test = utils.dataset.split_train_test(URM.tocoo(), validation_split)
            self.URM_test = URM_test.tocsr()
            super(SLIM_BPR_Cython, self).__init__(URM_train.tocsr(), ICM, exclude_seen)
        else:
            super(SLIM_BPR_Cython, self).__init__(URM, ICM, exclude_seen)

        self.epochs = epochs
        self.topK = topK
        self.learning_rate = learning_rate
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.batch_size = batch_size
        self.symmetric = symmetric
        self.sgd_mode = sgd_mode
        self.verbose = verbose
        self.gamma = gamma
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.patience = patience

        self.feature_weighting = feature_weighting

        self.n_users = URM.shape[0]
        self.n_items = URM.shape[1]

        self.train_with_sparse_weights = False

        if add_side_info:
            self.add_side_information(omega)

    def fit(self, verbose=True):
        self.URM = apply_feature_weighting(self.URM, self.feature_weighting)

        # URM matrix is 0-1s already
        URM_train_positive = self.URM.copy()

        self.cythonEpoch = SLIM_BPR_Cython_Epoch(URM_train_positive,
                                                 train_with_sparse_weights=self.train_with_sparse_weights,
                                                 final_model_sparse_weights=True,
                                                 topK=self.topK,
                                                 learning_rate=self.learning_rate,
                                                 li_reg=self.lambda_i,
                                                 lj_reg=self.lambda_j,
                                                 batch_size=1,
                                                 symmetric=self.symmetric,
                                                 sgd_mode=self.sgd_mode,
                                                 verbose=verbose,
                                                 random_seed=None,
                                                 gamma=self.gamma,
                                                 beta_1=self.beta_1,
                                                 beta_2=self.beta_2)

        # MAIN LOOP of training
        convergence = False
        best_MAP = 0
        epochs_current = 0
        lower_epochs = 0
        while epochs_current < self.epochs and not convergence:
            # run an epoch
            self.cythonEpoch.epochIteration_Cython()
            if self.patience != None:
                # prepare for validation
                self.get_S_set_W_set_predicted_URM()

                _, _, MAP = utils.evaluate.evaluate_algorithm(self.URM_test, self)
                if MAP > best_MAP:  # best model so far
                    print("Found new best MAP! Epoch:{}, New MAP:{}, Old MAP:{}".format(epochs_current, MAP, best_MAP))
                    self.S_best = self.S_incremental.copy()
                    best_MAP = MAP
                    lower_epochs = 0
                else:  # one more run without improvements
                    lower_epochs += 1
                    if lower_epochs > self.patience:
                        convergence = True

            epochs_current += 1

        if self.patience != None:
            # Restore best model so far:
            self.use_Sbest_set_W_set_predicted_URM()
        else:
            self.get_S_set_W_set_predicted_URM()

        self.cythonEpoch._dealloc()

        sys.stdout.flush()

    def get_S_set_W_set_predicted_URM(self):
        self.S_incremental = self.cythonEpoch.get_S()

        if self.train_with_sparse_weights:
            self.W_sparse = self.S_incremental
            self.W_sparse = check_matrix(self.W_sparse, format='csr')
        else:
            self.W_sparse = similarityMatrixTopK(self.S_incremental, k=self.topK)
            self.W_sparse = check_matrix(self.W_sparse, format='csr')

        # Precompute URM
        self.predicted_URM = self.URM.dot(self.W_sparse)

    def use_Sbest_set_W_set_predicted_URM(self):
        if self.train_with_sparse_weights:
            self.W_sparse = self.S_best
            self.W_sparse = check_matrix(self.W_sparse, format='csr')
        else:
            self.W_sparse = similarityMatrixTopK(self.S_best, k=self.topK)
            self.W_sparse = check_matrix(self.W_sparse, format='csr')

        # Precompute URM
        self.predicted_URM = self.URM.dot(self.W_sparse)