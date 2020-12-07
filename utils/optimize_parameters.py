import itertools
import os
import pandas as pd

import skopt
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

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
from recommenders.p3alpha import P3alphaRecommender, RP3betaRecommender

names = {}
spaces = {}

names[CBFRecommender] = "CBFRecommender__kcross" #Done and fixed
spaces[CBFRecommender] = [
    Integer(1, 2000, name='topK'),
    Real(1, 500, name='shrink'),
    Categorical([True, False], name='normalize'),
    Categorical(["cosine", "jaccard", 'tanimoto', 'dice'], name='similarity'),
    Categorical(["BM25"], name='feature_weighting'),
    Real(0.1, 200, name='K'),
    Real(0.01, 1, name='B'),
]

names[UserBasedCFRecommender] = "UserBasedCFRecommender_kcross_3" #TODO
spaces[UserBasedCFRecommender] = [
    Integer(1, 500, name='topK'),
    Real(1, 50, name='shrink'),
    Categorical([True], name='normalize'),
    Categorical(['tanimoto'], name='similarity'),
    Categorical([None, "TF-IDF"], name='feature_weighting')
]

names[ItemBasedCFRecommender] = "ItemBasedCFRecommender_kcross_2" #Done and fixed
spaces[ItemBasedCFRecommender] = [
    Integer(4500, 10000, name='topK'),
    Real(300, 1000, name='shrink'),
    Categorical([True], name='normalize'),
    Categorical(["cosine"], name='similarity'),
    Categorical(["TF-IDF"], name='feature_weighting'),
    #Real(0.1, 200, name='K'),
    #Real(0.1, 1, name='B'),
]

names[SLIM_BPR_Cython] = "SLIM_BPR_Cython_kcross" # Done
spaces[SLIM_BPR_Cython] = [
    Integer(1, 400, name='topK'),
    Categorical([1e-4, 1e-3, 1e-2], name="learning_rate"),
    Real(0, 2, name='lambda_i'),
    Real(0, 2, name='lambda_j'),
    Categorical([False], name='symmetric'),
    Categorical([200], name='epochs'), #Integer(1, 200, name="epochs")
]

names[SSLIMRMSERecommender] = "SSLIMRMSERecommender_kcross"
spaces[SSLIMRMSERecommender] = [
    Integer(50, 100, name='epochs'),
    Real(0, 3, name='beta'),
    Real(1e-5, 1e-3, name='l1_reg'),
    Categorical([1e-3, 1e-4, 1e-5], name='learning_rate'),
    Categorical([True], name='add_side_info'),
]

names[SVDRecommender] = "SVDRecommender_kcross" # Done
spaces[SVDRecommender] = [
    Integer(100, 600, name='latent_factors'),
    Categorical([True], name='scipy'),
]

names[ALSMFRecommender] = "ALSMFRecommender"
spaces[ALSMFRecommender] = [
    Real(0, 100, name='alpha'),
    Real(0, 15, name='lambda_val'),
]
names[ImplicitALSRecommender] = "ImplicitALSRecommender_kcross_2" #TODO
spaces[ImplicitALSRecommender] = [
    Integer(100, 800, name='latent_factors'),
    Real(0, 20, name='lambda_val'),
    Real(0, 100, name='alpha'),
]

names[LightFMRecommender] = "LightFMRecommender"
spaces[LightFMRecommender] = [
    Categorical(["warp", "bpr"], name="loss"),
    Categorical([1e-3, 1e-4, 1e-5, 1e-6], name='item_alpha'),
    Categorical([1e-3, 1e-4, 1e-5, 1e-6], name='user_alpha'),
    Integer(200, 400, name="num_components"),
    #Integer(20, 50, name="epochs"), #FIXED TO 30 due to time constraints
    Categorical([None, "BM25", "TF-IDF"], name='feature_weighting')
]

names[P3alphaRecommender] = "P3alphaRecommender_kcross_2" # Discarded in favor of RP3Beta :)
spaces[P3alphaRecommender] = [
    Integer(400, 600, name="topK"),
    Real(0, 1, name='alpha'),
    Categorical([True,False], name="normalize_similarity"),
    Categorical([None, "TF-IDF"], name='feature_weighting')
    #Real(0.1, 200, name='K'),
    #Real(0.01, 1, name='B'),
]


names[RP3betaRecommender] = "RP3betaRecommender_kcross2" #Done and fixed
spaces[RP3betaRecommender] = [
    Integer(600, 1200, name="topK"),
    Real(0, 1, name='alpha'),
    Real(0, 1, name='beta'),
    Categorical([False], name="normalize_similarity"),
    Categorical(["TF-IDF"], name='feature_weighting')
]

# names[HybridRecommender] = "HybridRecommender"
# spaces[HybridRecommender] = [
#     Categorical([0.0], name="TopPopweight"),
#     Real(0, 5, name='IBCFweight'),
#     Real(0, 5, name='UBCFweight'),
#     Real(0, 5, name='CBFweight'),
#     Categorical([0.0], name="SSLIMweight"),
#     Real(0, 5, name='ALSweight'),
#     Categorical([0.0], name="LFMCFweight"),
#     Real(0, 5, name='SLIMBPRweight'),
#     Categorical([0.0], name="SVDweight"),
#     Real(0, 1, name='P3weight'),
#     Categorical([True], name="normalize")
# ]

names[HybridRecommenderWithTopK] = "HybridRecommenderWithTopK_kcross_mid_yes_normalize_small_k"
spaces[HybridRecommenderWithTopK] = [
#    Real(0, 1, name='TopPopweight'),
    Real(0, 5, name='IBCFweight'),
    Real(0, 5, name='UBCFweight'),
    Real(0, 5, name='CBFweight'),
    Real(0, 5, name="SSLIMweight"),
    Real(0, 5, name='ALSweight'),
    Categorical([0.0], name="LFMCFweight"),
    Real(0, 5, name='SLIMBPRweight'),
    Real(0, 5, name="SVDweight"),
    Real(0, 5, name='P3weight'),
    Real(0, 5, name='RP3weight'),
    Categorical([True], name="normalize")
]


def load_df(name):
    filename = "./optimization_data/" + name + ".res"
    if os.path.exists(filename):
        df = pd.read_pickle(filename)
        return df
    else:
        return None


def read_df(name, param_names, metric="MAP"):
    df = load_df(name)

    if df is not None:
        y = df[metric].tolist()
        x_series = [df[param_name].tolist() for param_name in param_names]
        x = [t for t in zip(*x_series)]

        return x, y
    else:
        return None, None


def store_df(name, df: pd.DataFrame):
    filename = "./optimization_data/" + name + ".res"
    df.to_pickle(filename)


def append_new_data_to_df(name, new_df):
    df: pd.DataFrame = load_df(name)
    df = df.append(new_df, ignore_index=True)
    store_df(name, df)


def create_df(param_tuples, param_names, value_list, metric="MAP"):
    df = pd.DataFrame(data=param_tuples, columns=param_names)
    df[metric] = value_list
    return df


def optimize_parameters(URMrecommender_class: type, n_calls=100, k=5, validation_percentage=0.05, n_random_starts=None, seed=None, limit_at=1000):
    if n_random_starts is None:
        n_random_starts = int(0.5 * n_calls)

    name = names[URMrecommender_class]
    space = spaces[URMrecommender_class]

    if validation_percentage > 0:
        print("Using randomized datasets. k={}, val_percentage={}".format(k, validation_percentage))
        URM_trains, URM_tests, ICM_trains = utils.dataset.give_me_randomized_k_folds_with_val_percentage(k, validation_percentage)
    else:
        print("Splitting original datasets in N_folds:{}".format(k))
        URM_trains, URM_tests, ICM_trains = utils.dataset.give_me_k_folds(k)

    if len(URM_trains) > limit_at:
        URM_trains = URM_trains[:limit_at]
        URM_tests = URM_tests[:limit_at]
        ICM_trains = ICM_trains[:limit_at]

    assert(len(URM_trains) == len(URM_tests) and len(URM_tests) == len(ICM_trains))
    print("Starting optimization: N_folds={}".format(len(URM_trains)))

    if URMrecommender_class == HybridRecommender or URMrecommender_class == HybridRecommenderWithTopK:
        recommenders = []

        for URM_train_csr, ICM_csr in zip(URM_trains, ICM_trains):
            recommenders.append(URMrecommender_class(URM_train_csr, ICM_csr))

        @use_named_args(space)
        def objective(**params):
            scores = []
            for recommender, test in zip(recommenders, URM_tests):
                recommender.fit(**params)
                _, _, MAP = utils.evaluate.evaluate_algorithm(test, recommender)
                scores.append(-MAP)
            print("Just Evaluated this: {}".format(params))
            return sum(scores) / len(scores)

    else:
        @use_named_args(space)
        def objective(**params):
            scores = []
            for URM_train_csr, ICM_csr, test in zip(URM_trains, ICM_trains, URM_tests):
                recommender = URMrecommender_class(URM_train_csr, ICM_csr, **params)
                recommender.fit()
                _, _, MAP = utils.evaluate.evaluate_algorithm(test, recommender)
                scores.append(-MAP)
            print("Just Evaluated this: {}".format(params))
            return sum(scores) / len(scores)

    # xs, ys = _load_xy(name)
    param_names = [v.name for v in spaces[URMrecommender_class]]
    xs, ys = read_df(name, param_names)

    res_gp = skopt.gp_minimize(objective, space, n_calls=n_calls, random_state=seed, x0=xs, y0=ys, verbose=True,
                               noise=1e-10, n_random_starts=n_random_starts
                               )

    print("Writing a total of {} points for {}. Newly added records: {}".format(len(res_gp.x_iters), name,
                                                                                n_calls))

    # _store_xy(name, res_gp.x_iters, res_gp.func_vals)
    df = create_df(res_gp.x_iters, param_names, res_gp.func_vals, "MAP")
    store_df(names[URMrecommender_class], df)

    print(name + " reached best performance = ", -res_gp.fun, " at: ", res_gp.x)


def grid_search(URM, ICM, URM_test, URMrecommender_class: type, dictionary_params: dict):
    names = dictionary_params.keys()
    lists = dictionary_params.values()

    iterable = itertools.product(*lists)

    df = pd.DataFrame()
    i = 0
    for tuple in iterable:
        params = {key: value for key, value in zip(names, tuple)}

        recommender = URMrecommender_class(URM, ICM, **params)
        recommender.fit()
        precision, recall, MAP = utils.evaluate.evaluate_algorithm(URM_test, recommender)
        params["MAP"] = MAP

        df = df.append(params, ignore_index=True)
        print("Grid Search evaluation {}. Params are {} and MAP is {}".format(i, params, MAP))
        i += 1

    append_new_data_to_df(names[URMrecommender_class], df)
