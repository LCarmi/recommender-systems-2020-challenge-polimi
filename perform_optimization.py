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
from recommenders.p3alpha import P3alphaRecommender, RP3betaRecommender


if __name__ == '__main__':

    val_percentage = 0
    k = 10
    n_calls = 200
    limit_at = 10

    rec_class = HybridRecommenderWithTopK
    utils.optimize_parameters.optimize_parameters(
        URMrecommender_class=rec_class,
        validation_percentage=val_percentage,
        k=k,
        n_calls=n_calls,
        limit_at=limit_at
    )
