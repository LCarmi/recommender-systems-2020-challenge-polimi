import pandas as pd
import numpy as np
import scipy
import os
from scipy import sparse as sps


def load_dataset(base_folder="./data"):
    def _load_dataset():
        ICM = pd.read_csv(os.path.join(base_folder, "data_ICM_title_abstract.csv"),
                          sep=",",
                          names=["row", "col", "data"],
                          header=0,
                          dtype={"row": np.int32, "col": np.int32, "data": np.float}
                          )
        targets = pd.read_csv(os.path.join(base_folder, "data_target_users_test.csv"),
                              sep=",",
                              names=["user_id"],
                              header=0,
                              dtype={"user_id": np.int32}
                              )
        URM = pd.read_csv(os.path.join(base_folder, "data_train.csv"),
                          sep=",",
                          names=["row", "col", "data"],
                          header=0,
                          dtype={"row": np.int32, "col": np.int32, "data": np.int32}
                          )

        return URM, ICM, targets

    URM, ICM, targets = _load_dataset()

    # NO NEED TO TRANSLATE
    # unique_users = URM["row"].unique()
    # unique_items = np.unique(np.concatenate((ICM["row"].unique(), URM["col"].unique()), axis=0))
    # unique_features = ICM["col"].unique()
    #
    # user_original_ID_to_index_dict = {}
    # item_original_ID_to_index_dict = {}
    # feature_original_ID_to_index_dict = {}
    # for user_id in unique_users:
    #     user_original_ID_to_index_dict[user_id] = len(user_original_ID_to_index_dict)
    # for item_id in unique_items:
    #     item_original_ID_to_index_dict[item_id] = len(item_original_ID_to_index_dict)
    # for feature_id in unique_features:
    #     feature_original_ID_to_index_dict[feature_id] = len(feature_original_ID_to_index_dict)
    #
    # URM["row"] = [user_original_ID_to_index_dict[user_original] for user_original in URM["row"].values]
    # URM["col"] = [item_original_ID_to_index_dict[item_original] for item_original in URM["col"].values]
    # ICM["row"] = [item_original_ID_to_index_dict[item_original] for item_original in ICM["row"].values]
    # ICM["col"] = [feature_original_ID_to_index_dict[feature_original] for feature_original in ICM["col"].values]

    URM_coo = sps.coo_matrix((URM["data"].values, (URM["row"].values, URM["col"].values)))
    ICM_coo = sps.coo_matrix((ICM["data"].values, (ICM["row"].values, ICM["col"].values)))
    targets = targets["user_id"].values

    return URM_coo, ICM_coo, targets


def split_train_test(URM_coo, percentage_validation):
    n_interactions = URM_coo.nnz

    train_mask = np.random.choice([True, False], n_interactions, p=[1 - percentage_validation, percentage_validation])
    URM_train_coo = sps.coo_matrix((URM_coo.data[train_mask], (URM_coo.row[train_mask], URM_coo.col[train_mask])), shape=URM_coo.shape)

    test_mask = np.logical_not(train_mask)
    URM_test_coo = sps.coo_matrix((URM_coo.data[test_mask], (URM_coo.row[test_mask], URM_coo.col[test_mask])), shape=URM_coo.shape)

    return URM_train_coo, URM_test_coo


def give_me_splitted_dataset(val_percentage):
    URM_coo, ICM_coo, targets = load_dataset(base_folder="./data")
    URM_train_coo, URM_test_coo = split_train_test(URM_coo, val_percentage)

    URM_train_csr = URM_train_coo.tocsr()
    URM_test_csr = URM_test_coo.tocsr()
    ICM_csr = ICM_coo.tocsr()

    return URM_train_csr, URM_test_csr, ICM_csr, targets
