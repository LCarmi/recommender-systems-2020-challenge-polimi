import time

import utils.dataset
import utils.evaluate
from recommenders.collaborativebasedfiltering import UserBasedCFRecommender, ItemBasedCFRecommender
from recommenders.contentbasedfiltering import CBFRecommender
from recommenders.hybrid import HybridRecommender, HybridRecommenderWithTopK
from recommenders.mf_ials import ALSMFRecommender, ImplicitALSRecommender
from recommenders.sslimrmse import SSLIMRMSERecommender
from recommenders.svd import SVDRecommender
from recommenders.test import RandomRecommender, TopPopRecommender
from recommenders.recommender import Recommender
from recommenders.slimbpr import SLIM_BPR_Cython
from recommenders.lightfm import LightFMRecommender
from recommenders.p3alpha import P3alphaRecommender


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

def run_scoring_k_fold(rec_class, val_pecentage, k):
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

if __name__ == '__main__':
    rec_class = HybridRecommenderWithTopK
    val_percentage = 0.2

    run_scoring(rec_class, val_percentage)
