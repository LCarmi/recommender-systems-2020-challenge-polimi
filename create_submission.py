import numpy as np

from recommenders import *



def prepare_submission(users_to_recommend: np.array, recommender: Recommender):
    recommendation_length = 10
    submission = []

    for user_id in users_to_recommend:
        recommendations = recommender.recommend(user_id=user_id,
                                                at=recommendation_length)
        submission.append((user_id, recommendations))

    return submission


def write_submission(submissions):
    with open("./submission.csv", "w+") as f:
        f.write("user_id,item_list\n")
        for user_id, items in submissions:
            f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")

def run_all_data_train():
    URM_coo, ICM_coo, targets = utils.dataset.load_dataset(base_folder="./data")
    URM_train_csr = URM_coo.tocsr()
    ICM_csr = ICM_coo.tocsr()

    recommender = HybridRecommender(URM_train_csr, ICM_csr)
    recommender.fit()

    submissions = prepare_submission(targets, recommender)
    write_submission(submissions)

if __name__ == '__main__':
    run_all_data_train()