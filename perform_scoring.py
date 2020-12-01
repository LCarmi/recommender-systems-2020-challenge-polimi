import time

import utils.dataset
import utils.evaluate
from recommenders import *


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
    rec_class = LightFMRecommender
    val_percentage = 0.2

    run_scoring(rec_class, val_percentage, num_components=400)
