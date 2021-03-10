import time

import utils.dataset
import utils.evaluate
from recommenders.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython
from recommenders.collaborativebasedfiltering import UserBasedCFRecommender, ItemBasedCFRecommender
from recommenders.contentbasedfiltering import CBFRecommender
from recommenders.hybrid import HybridRecommender, HybridRecommenderWithTopK, SequentialHybrid, EnrichingHybrid
from recommenders.mf_ials import ALSMFRecommender, ImplicitALSRecommender
from recommenders.sslimrmse import SSLIMRMSERecommender, SSLIMElasticNetRecommender
from recommenders.svd import SVDRecommender
from recommenders.test import RandomRecommender, TopPopRecommender
from recommenders.recommender import Recommender
from recommenders.slimbpr import SLIM_BPR_Cython
from recommenders.lightfm import LightFMRecommender
from recommenders.graphbased import P3alphaRecommender, RP3betaRecommender


def run_scoring(rec_class, val_percentage, **kwargs):
    # Data Loading
    URM_train_csr, URM_test_csr, ICM_csr, targets = utils.dataset.give_me_splitted_dataset(val_percentage)

    recommender = rec_class(URM_train_csr, ICM_csr, **kwargs)
    recommender.fit()

    start = time.time()
    precision, recall, MAP = utils.evaluate.evaluate_algorithm(URM_test_csr, recommender)
    end = time.time()

    print(
        "Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}. Time for full recommendation: {}".format(
            precision, recall, MAP, end - start))


def run_scoring_k_fold(rec_class, val_percentage, k):
    if val_percentage == 0.0:
        URM_trains, URM_tests, ICM_trains = utils.dataset.give_me_k_folds(k)
    else:
        URM_trains, URM_tests, ICM_trains = utils.dataset.give_me_randomized_k_folds_with_val_percentage(k,
                                                                                                         val_percentage)

    maps = []
    for i in range(k):
        recommender = rec_class(URM_trains[i], ICM_trains[i])
        recommender.fit()

        start = time.time()
        precision, recall, MAP = utils.evaluate.evaluate_algorithm(URM_tests[i], recommender)
        end = time.time()

        print(
            "Iteration {} ended. Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}. Time for full recommendation: {}".format(
                i, precision, recall, MAP, end - start))
        maps.append(MAP)

    print("Evaluation ended. Mean MAP is: {}".format(sum(maps) / len(maps)))


def test_gd_on_hybrid():
    URM_train_csr, URM_test_csr, ICM_csr, targets = utils.dataset.give_me_splitted_dataset(0.2)
    rec = HybridRecommenderWithTopK(URM_train_csr, ICM_csr)
    rec.fit()
    print(utils.evaluate.evaluate_algorithm(URM_test_csr, rec))

    rec.train(100)
    print(utils.evaluate.evaluate_algorithm(URM_test_csr, rec))


def test_single_rec():
    rec_class = SVDRecommender

    run_scoring_k_fold(rec_class, val_percentage=0.0, k=10)


if __name__ == '__main__':
    test_single_rec()
