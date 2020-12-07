import time

import utils.dataset
import utils.evaluate
from recommenders.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython
from recommenders.collaborativebasedfiltering import UserBasedCFRecommender, ItemBasedCFRecommender
from recommenders.contentbasedfiltering import CBFRecommender
from recommenders.hybrid import HybridRecommender, HybridRecommenderWithTopK
from recommenders.mf_ials import ALSMFRecommender, ImplicitALSRecommender
from recommenders.sslimrmse import SSLIMRMSERecommender, SSLIMElasticNetRecommender
from recommenders.svd import SVDRecommender
from recommenders.test import RandomRecommender, TopPopRecommender
from recommenders.recommender import Recommender
from recommenders.slimbpr import SLIM_BPR_Cython
from recommenders.lightfm import LightFMRecommender
from recommenders.p3alpha import P3alphaRecommender, RP3betaRecommender


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
    URM_coo, ICM_coo, targets = utils.dataset.load_dataset(base_folder="./data")
    ICM_csr = ICM_coo.tocsr()

    maps = []
    for i in range(k):
        URM_train_coo, URM_test_coo = utils.dataset.split_train_test(URM_coo, val_percentage)
        URM_train_csr = URM_train_coo.tocsr()
        URM_test_csr = URM_test_coo.tocsr()

        recommender = rec_class(URM_train_csr, ICM_csr)
        recommender.fit()

        start = time.time()
        precision, recall, MAP = utils.evaluate.evaluate_algorithm(URM_test_csr, recommender)
        end = time.time()

        print(
            "Iteration {} ended. Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}. Time for full recommendation: {}".format(
                i, precision, recall, MAP, end - start))
        maps.append(MAP)

    print("Evaluation ended. Mean MAP is: {}".format(sum(maps)/len(maps)))


def test_gd_on_hybrid():
    URM_train_csr, URM_test_csr, ICM_csr, targets = utils.dataset.give_me_splitted_dataset(0.2)
    rec = HybridRecommenderWithTopK(URM_train_csr, ICM_csr)
    rec.fit()
    print(utils.evaluate.evaluate_algorithm(URM_test_csr, rec))

    rec.train(100)
    print(utils.evaluate.evaluate_algorithm(URM_test_csr, rec))

def test_single_rec():
    rec_class = SLIM_BPR_Cython
    val_percentage = 0.10

    run_scoring_k_fold(rec_class, val_percentage, k=3)


if __name__ == '__main__':
    test_single_rec()



