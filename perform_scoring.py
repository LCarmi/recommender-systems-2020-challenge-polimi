import time

import utils.dataset
import utils.evaluate
from recommenders.collaborativebasedfiltering import UserBasedCFRecommender, ItemBasedCFRecommender
from recommenders.contentbasedfiltering import CBFRecommender
from recommenders.hybrid import HybridRecommender
from recommenders.mf_ials import ALSMFRecommender
from recommenders.sslimrmse import SSLIMRMSERecommender
from recommenders.svd import SVDRecommender
from recommenders.test import RandomRecommender, TopPopRecommender
from recommenders.recommender import Recommender
from recommenders.slimbpr import SLIM_BPR_Cython
from recommenders.lightfm import LightFMRecommender
from recommenders.p3alpha import P3alphaRecommender


def run_scoring(rec_class, val_percentage, **kwargs):
    # Data Loading
    URM_coo, ICM_coo, targets = utils.dataset.load_dataset(base_folder="./data")
    URM_train_coo, URM_test_coo = utils.dataset.split_train_test(URM_coo, val_percentage)

    URM_train_csr = URM_train_coo.tocsr()
    URM_test_csr = URM_test_coo.tocsr()
    ICM_csr = ICM_coo.tocsr()

    recommender = rec_class(URM_train_csr, ICM_csr, **kwargs)
    recommender.fit()

    start = time.time()
    precision, recall, MAP = utils.evaluate.evaluate_algorithm(URM_test_csr, recommender)
    end = time.time()

    print(
        "Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}. Time for full recommendation: {}".format(
            precision, recall, MAP, end - start))


if __name__ == '__main__':
    rec_class = P3alphaRecommender
    val_percentage = 0.2

    run_scoring(rec_class, val_percentage)
