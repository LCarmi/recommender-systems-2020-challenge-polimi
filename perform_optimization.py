import utils.dataset
import utils.optimize_parameters
from recommenders import *


def run_optimization(rec_class, n_calls, val_percentage):
    # Data Loading
    URM_coo, ICM_coo, targets = utils.dataset.load_dataset(base_folder="./data")
    URM_train_coo, URM_test_coo = utils.dataset.split_train_test(URM_coo, val_percentage)

    URM_train_csr = URM_train_coo.tocsr()
    URM_test_csr = URM_test_coo.tocsr()
    ICM_csr = ICM_coo.tocsr()

    utils.optimize_parameters.optimize_parameters(URM_train_csr, ICM_csr, URM_test_csr, rec_class, n_calls)


if __name__ == '__main__':
    rec_class = LightFMRecommender
    n_calls = 50
    val_percentage = 0.2

    run_optimization(rec_class, n_calls, val_percentage)
