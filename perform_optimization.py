import utils.dataset
import utils.optimize_parameters

from recommenders.collaborativebasedfiltering import *
from recommenders.contentbasedfiltering import *
from recommenders.hybrid import *
from recommenders.mf_ials import *
from recommenders.sslimrmse import *
from recommenders.svd import *
from recommenders.test import *
from recommenders.slimbpr import *
from recommenders.lightfm import *
from recommenders.graphbased import *


if __name__ == '__main__':

    val_percentage = 0
    k = 10
    limit_at = 10

    rec_class = HybridRecommenderWithTopK
    utils.optimize_parameters.optimize_parameters(
        URMrecommender_class=rec_class,
        validation_percentage=val_percentage,
        n_random_starts=250,
        k=k,
        n_calls=1000,
        limit_at=limit_at,
        forest=True,
        xi=0.001
    )

    # rec_class = SVDRecommender
    # utils.optimize_parameters.optimize_parameters(
    #     URMrecommender_class=rec_class,
    #     validation_percentage=val_percentage,
    #     k=k,
    #     n_calls=n_calls,
    #     limit_at=limit_at
    # )

    # rec_class = UserBasedCFRecommender
    # utils.optimize_parameters.optimize_parameters(
    #     URMrecommender_class=rec_class,
    #     validation_percentage=val_percentage,
    #     k=k,
    #     n_calls=100,
    #     limit_at=limit_at,
    #     forest=True,
    # )

    # rec_class = ItemBasedCFRecommender
    # utils.optimize_parameters.optimize_parameters(
    #     URMrecommender_class=rec_class,
    #     validation_percentage=val_percentage,
    #     k=k,
    #     n_calls=100,
    #     limit_at=limit_at,
    #     forest=True,
    # )

    # rec_class = CBFRecommender
    # utils.optimize_parameters.optimize_parameters(
    #     URMrecommender_class=rec_class,
    #     validation_percentage=val_percentage,
    #     k=k,
    #     n_calls=n_calls,
    #     limit_at=limit_at
    # )

    # rec_class = SLIM_BPR_Cython
    # utils.optimize_parameters.optimize_parameters(
    #     URMrecommender_class=rec_class,
    #     validation_percentage=val_percentage,
    #     k=10,
    #     n_calls=50,
    #     limit_at=10,
    #     forest=True
    # )
    #
    # rec_class = P3alphaRecommender
    # utils.optimize_parameters.optimize_parameters(
    #     URMrecommender_class=rec_class,
    #     validation_percentage=val_percentage,
    #     k=k,
    #     limit_at=limit_at,
    #     forest=True,
    # )
    #
    # rec_class = P3alphaRecommenderSI
    # utils.optimize_parameters.optimize_parameters(
    #     URMrecommender_class=rec_class,
    #     validation_percentage=val_percentage,
    #     k=k,
    #     limit_at=limit_at,
    #     forest=True,
    # )
    #
    # rec_class = RP3betaRecommender
    # utils.optimize_parameters.optimize_parameters(
    #     URMrecommender_class=rec_class,
    #     validation_percentage=val_percentage,
    #     k=k,
    #     n_calls=100,
    #     limit_at=limit_at,
    #     forest=True,
    # )
