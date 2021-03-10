import itertools
import os
import pandas as pd
import skopt
from skopt.utils import use_named_args

import utils.evaluate
from utils.hyperparam_def import names, spaces
from recommenders.hybrid import *


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


def optimize_parameters(URMrecommender_class: type, n_calls=100, k=5, validation_percentage=0.05, n_random_starts=None,
                        seed=None, limit_at=1000, forest=False, xi=0.01):
    if n_random_starts is None:
        n_random_starts = int(0.5 * n_calls)

    name = names[URMrecommender_class]
    space = spaces[URMrecommender_class]

    if validation_percentage > 0:
        print("Using randomized datasets. k={}, val_percentage={}".format(k, validation_percentage))
        URM_trains, URM_tests, ICM_trains = utils.dataset.give_me_randomized_k_folds_with_val_percentage(k,
                                                                                                         validation_percentage)
    else:
        print("Splitting original datasets in N_folds:{}".format(k))
        URM_trains, URM_tests, ICM_trains = utils.dataset.give_me_k_folds(k)

    if len(URM_trains) > limit_at:
        URM_trains = URM_trains[:limit_at]
        URM_tests = URM_tests[:limit_at]
        ICM_trains = ICM_trains[:limit_at]

    assert (len(URM_trains) == len(URM_tests) and len(URM_tests) == len(ICM_trains))
    print("Starting optimization: N_folds={}, name={}".format(len(URM_trains), names[URMrecommender_class]))

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

    if not forest:
        res_gp = skopt.gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            n_points=10000,
            n_jobs=1,
            # noise = 'gaussian',
            noise=1e-5,
            acq_func='gp_hedge',
            acq_optimizer='auto',
            random_state=None,
            verbose=True,
            n_restarts_optimizer=10,
            xi=xi,
            kappa=1.96,
            x0=xs,
            y0=ys,
        )
    else:
        res_gp = skopt.forest_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            verbose=True,
            x0=xs,
            y0=ys,
            acq_func="EI",
            xi=xi
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
