import utils.dataset
import utils.optimize_parameters

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


def run_optimization(rec_class, n_calls, val_percentage):
    # Data Loading
    URM_coo, ICM_coo, targets = utils.dataset.load_dataset(base_folder="./data")
    URM_train_coo, URM_test_coo = utils.dataset.split_train_test(URM_coo, val_percentage)

    URM_train_csr = URM_train_coo.tocsr()
    URM_test_csr = URM_test_coo.tocsr()
    ICM_csr = ICM_coo.tocsr()

    utils.optimize_parameters.optimize_parameters(URM_train_csr, ICM_csr, URM_test_csr, rec_class, n_calls)


if __name__ == '__main__':


    val_percentage = 0.2
    rec_class = HybridRecommenderWithTopK
    n_calls = 200
    utils.optimize_parameters.optimize_parameters(None, None, None, rec_class, n_calls)
    #run_optimization(rec_class, n_calls, val_percentage)
