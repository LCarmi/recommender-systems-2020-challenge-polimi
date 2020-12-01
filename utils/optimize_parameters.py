import itertools
import os
import pandas as pd

import skopt
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

import utils.evaluate
from recommenders import *

names = {}
spaces = {}

names[CBFRecommender] = "CBFRecommender"
spaces[CBFRecommender] = [
    Integer(1, 2000, name='topK'),
    Real(1, 500, name='shrink'),
    Categorical([True, False], name='normalize'),
    Categorical(["cosine", "pearson", "adjusted", "jaccard", 'tanimoto', 'dice'], name='similarity'),
    Categorical([None, "BM-25", "TF-IDF"], name='feature_weighting')
]

names[UserBasedCFRecommender] = "UserBasedCFRecommender"
spaces[UserBasedCFRecommender] = [
    Integer(1, 2000, name='topK'),
    Real(1, 300, name='shrink'),
    Categorical([True], name='normalize'),
    Categorical(["cosine", "pearson", "adjusted", "jaccard", 'tanimoto', 'dice'], name='similarity'),
    Categorical([None, "BM-25", "TF-IDF"], name='feature_weighting')
]

names[ItemBasedCFRecommender] = "ItemBasedCFRecommender"
spaces[ItemBasedCFRecommender] = [
    Integer(1, 5000, name='topK'),
    Real(1, 500, name='shrink'),
    Categorical([True, False], name='normalize'),
    Categorical(["cosine", "pearson", "adjusted", "asymmetric", "jaccard", 'tanimoto', 'dice'], name='similarity'),
    Categorical([None, "BM-25", "TF-IDF"], name='feature_weighting')
]

names[SLIM_BPR_Cython] = "SLIM_BPR_Cython"
spaces[SLIM_BPR_Cython] = [
    Integer(1, 2000, name='topK'),
    Categorical([1e-4, 1e-3, 1e-2], name="learning_rate"),
    Real(0, 1, name='lambda_i'),
    Real(0, 1, name='lambda_j'),
    Categorical([True, False], name='symmetric'),
    Integer(1, 500, name="epochs")
]

names[SSLIMRMSERecommender] = "SSLIMRMSERecommender"
spaces[SSLIMRMSERecommender] = [
    Integer(1, 300, name='epochs'),
    Real(0, 1, name='beta'),
    Categorical([1e-2, 1e-3, 1e-4, 1e-5], name='learning_rate'),
    Categorical([True, False], name='add_side_info'),
]

names[SVDRecommender] = "SVDRecommender"
spaces[SVDRecommender] = [
    Integer(1, 600, name='latent_factors'),
    Categorical([True, False], name='scipy'),
]

names[ALSMFRecommender] = "ALSMFRecommender"
spaces[ALSMFRecommender] = [
    Real(0, 100, name='alpha'),
    Real(0, 5, name='lambda_val'),
]

names[HybridRecommender] = "HybridRecommender"
spaces[HybridRecommender] = [
    # IBCFweight=1, UBCFweight=1, LFMCFweight=0.0, CBFweight=1, SSLIMweight=0.0, ALSweight=0.0, SLIMBPRweight=0.0
    Real(0, 1, name='IBCFweight'),
    Real(0, 1, name='UBCFweight'),
    Real(0, 1, name='CBFweight'),
    Real(0, 1, name='SSLIMweight'),
    Real(0, 1, name='ALSweight'),
    Real(0, 1, name='LFMCFweight'),
    # REFERENCE #Real(0, 5, name='SLIMBPRweight'),
    Real(0, 1, name='SVDweight')
]

names[LightFMRecommender] = "LightFMRecommender"
spaces[LightFMRecommender] = [
    Categorical(["warp", "bpr"], name="loss"),
    Categorical([1e-3, 1e-4, 1e-5, 1e-6], name='item_alpha'),
    Categorical([1e-3, 1e-4, 1e-5, 1e-6], name='user_alpha'),
    Integer(200, 400, name="num_components"),
    #Integer(20, 50, name="epochs"), #FIXED TO 30 due to time constraints
    Categorical([None, "BM-25", "TF-IDF"], name='feature_weighting')
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


def optimize_parameters(URM, ICM, URM_test, URMrecommender_class: type, n_calls=100, n_random_starts=None, seed=None):
    if n_random_starts is None:
        n_random_starts = int(0.5 * n_calls)

    name = names[URMrecommender_class]
    space = spaces[URMrecommender_class]

    if URMrecommender_class is HybridRecommender:
        recommender = HybridRecommender(URM, ICM)
        recommender.fit()

        @use_named_args(space)
        def objective(**params):
            recommender.set_weights(**params)
            _, _, MAP = utils.evaluate.evaluate_algorithm(URM_test, recommender)
            return -MAP
    else:
        @use_named_args(space)
        def objective(**params):
            recommender = URMrecommender_class(URM, ICM, **params)
            recommender.fit()
            _, _, MAP = utils.evaluate.evaluate_algorithm(URM_test, recommender)
            return -MAP

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
