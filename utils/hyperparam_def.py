from skopt.space import Real, Integer, Categorical

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

names = {}
spaces = {}

names[CBFRecommender] = "CBFRecommender_final"  # Done and fixed
spaces[CBFRecommender] = [
    Integer(1, 2000, name='topK'),
    Real(1, 500, name='shrink'),
    Categorical([True, False], name='normalize'),
    Categorical(["cosine", "jaccard", 'tanimoto', 'dice'], name='similarity'),
    Categorical([None, "TF-IDF", "TF-IDF-Transpose", "BM25", "BM25-Transpose"], name='feature_weighting'),
    Real(0.1, 200, name='K'),
    Real(0.01, 1, name='B'),
]

names[UserBasedCFRecommender] = "UserBasedCFRecommender_tree"
spaces[UserBasedCFRecommender] = [
    Integer(1, 500, name='topK'),
    Real(1, 50, name='shrink'),
    Categorical([True], name='normalize'),
    Categorical(['tanimoto'], name='similarity'),
    Categorical([None, "TF-IDF", "TF-IDF-Transpose"], name='feature_weighting'),
    #Categorical([None, "TF-IDF", "TF-IDF-Transpose", "BM25", "BM25-Transpose"], name='feature_weighting'),
    #Real(0.1, 50, name='K'),
    #Real(0.01, 1, name='B'),
]

names[UserBasedCFRecommenderSI] = "UserBasedCFRecommender_tree_side_info"
spaces[UserBasedCFRecommenderSI] = [
    Integer(1, 500, name='topK'),
    Real(1, 50, name='shrink'),
    Categorical([True], name='normalize'),
    Categorical(['tanimoto'], name='similarity'),
    Categorical([None, "TF-IDF", "TF-IDF-Transpose"], name='feature_weighting'),
    Real(0, 50, name='omega')
]

names[ItemBasedCFRecommender] = "ItemBasedCFRecommender_tree"
spaces[ItemBasedCFRecommender] = [
    Integer(0, 10000, name='topK'),
    Real(0, 1000, name='shrink'),
    Categorical([True], name='normalize'),
    Categorical(["cosine", "tanimoto", "dice"], name='similarity'),
    Categorical([None, "TF-IDF", "TF-IDF-Transpose"], name='feature_weighting')
]

names[ItemBasedCFRecommenderSI] = "ItemBasedCFRecommender_tree_side_info"
spaces[ItemBasedCFRecommenderSI] = [
    Integer(0, 10000, name='topK'),
    Real(0, 1000, name='shrink'),
    Categorical([True], name='normalize'),
    Categorical(["cosine", "tanimoto", "dice"], name='similarity'),
    Categorical([None, "TF-IDF", "TF-IDF-Transpose"], name='feature_weighting'),
    Real(0, 50, name='omega'),
]

names[SLIM_BPR_Cython] = "SLIM_BPR_Cython_tree_side_info"
spaces[SLIM_BPR_Cython] = [
    Integer(1, 400, name='topK'),
    Categorical([1e-4, 1e-3, 1e-2], name="learning_rate"),
    Real(0, 2, name='lambda_i'),
    Real(0, 2, name='lambda_j'),
    Categorical([False], name='symmetric'),
    Categorical([150], name='epochs'),  # Integer(1, 200, name="epochs")
    Categorical([None, "TF-IDF", "TF-IDF-Transpose"], name='feature_weighting'),
    Real(0, 50, name='omega')
]

names[SSLIMRMSERecommender] = "SSLIMRMSERecommender_kcross"
spaces[SSLIMRMSERecommender] = [
    Integer(50, 100, name='epochs'),
    Real(0, 3, name='beta'),
    Real(1e-5, 1e-3, name='l1_reg'),
    Real(1e-5, 1e-3, name='learning_rate'),
    Categorical([True], name='add_side_info'),
]

names[SVDRecommenderSI] = "SVDRecommender_tree_side_info"
spaces[SVDRecommenderSI] = [
    Integer(100, 600, name='latent_factors'),
    Categorical([True], name='scipy'),
    Categorical([None, "TF-IDF", "TF-IDF-Transpose"], name='feature_weighting'),
    Real(0, 50, name='omega')
]

names[ALSMFRecommender] = "ALSMFRecommender"
spaces[ALSMFRecommender] = [
    Real(0, 100, name='alpha'),
    Real(0, 15, name='lambda_val'),
]
names[ImplicitALSRecommender] = "ImplicitALSRecommender_tree_side_info"
spaces[ImplicitALSRecommender] = [
    Integer(100, 800, name='latent_factors'),
    Real(0, 20, name='lambda_val'),
    Real(0, 50, name='alpha'),
    Categorical([None, "TF-IDF", "TF-IDF-Transpose"], name='feature_weighting'),
    Real(0, 50, name='omega')
    # Categorical([None, "TF-IDF", "TF-IDF-Transpose", "BM25", "BM25-Transpose"], name='feature_weighting'),
    # Real(1, 20, name='K'),
    # Real(0.5, 1, name='B'),
]

names[LightFMRecommender] = "LightFMRecommender"
spaces[LightFMRecommender] = [
    Categorical(["warp", "bpr"], name="loss"),
    Categorical([1e-3, 1e-4, 1e-5, 1e-6], name='item_alpha'),
    Categorical([1e-3, 1e-4, 1e-5, 1e-6], name='user_alpha'),
    Integer(200, 400, name="num_components"),
    # Integer(20, 50, name="epochs"), #FIXED TO 30 due to time constraints
    Categorical([None, "BM25", "TF-IDF"], name='feature_weighting')
]

names[P3alphaRecommender] = "P3alphaRecommender_tree"
spaces[P3alphaRecommender] = [
    Integer(400, 600, name="topK"),
    Real(0, 1, name='alpha'),
    Categorical([True, False], name="normalize_similarity"),
    Categorical([None, "TF-IDF", "TF-IDF-Transpose", "BM25", "BM25-Transpose"], name='feature_weighting'),
    Real(1, 20, name='K'),
    Real(0.5, 1, name='B'),
]

names[P3alphaRecommenderSI] = "P3alphaRecommender_tree_side_info"
spaces[P3alphaRecommenderSI] = [
    Integer(400, 600, name="topK"),
    Real(0, 1, name='alpha'),
    Categorical([False], name="normalize_similarity"),
    Categorical([None, "TF-IDF", "TF-IDF-Transpose", "BM25", "BM25-Transpose"], name='feature_weighting'),
    Real(1, 20, name='K'),
    Real(0.5, 1, name='B'),
]

names[RP3betaRecommender] = "RP3betaRecommender_tree"
spaces[RP3betaRecommender] = [
    Integer(400, 1200, name="topK"),
    Real(0, 1, name='alpha'),
    Real(0, 1, name='beta'),
    Categorical([False], name="normalize_similarity"),
    Categorical([None, "TF-IDF", "TF-IDF-Transpose", "BM25", "BM25-Transpose"], name='feature_weighting'),
    Real(1, 20, name='K'),
    Real(0.5, 1, name='B'),
]

names[RP3betaRecommenderSI] = "RP3betaRecommender_tree_side_info"
spaces[RP3betaRecommenderSI] = [
    Integer(400, 1200, name="topK"),
    Real(0, 1, name='alpha'),
    Real(0, 1, name='beta'),
    Categorical([False], name="normalize_similarity"),
    Categorical([None, "TF-IDF", "TF-IDF-Transpose"], name='feature_weighting'),
    Real(0, 50, name='omega')
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

# names[HybridRecommenderWithTopK] = "HybridRecommenderWithTopK_kcross_mid_yes_normalize_small_k"
# spaces[HybridRecommenderWithTopK] = [
#     #    Real(0, 1, name='TopPopweight'),
#     Real(0, 5, name='IBCFweight'),
#     Real(0, 5, name='UBCFweight'),
#     Real(0, 5, name='CBFweight'),
#     Real(0, 5, name="SSLIMweight"),
#     Real(0, 5, name='ALSweight'),
#     Categorical([0.0], name="LFMCFweight"),
#     Real(0, 5, name='SLIMBPRweight'),
#     Real(0, 5, name="SVDweight"),
#     Real(0, 5, name='P3weight'),
#     Real(0, 5, name='RP3weight'),
#     Categorical([True], name="normalize")
# ]

# names[HybridRecommenderWithTopK] = "HybridRecommenderWithTopK_tree_side_info_normalized_small"
# spaces[HybridRecommenderWithTopK] = [
#     Real(0.0001, 5, name='TopPopweight'),
#     Real(0, 5, name='IBCFweight'), # Actually 0
#     Real(0, 5, name='UBCFweight'), # Actually 0
#     Real(0, 5, name='IBCFweightSI'),
#     Real(0, 5, name='UBCFweightSI'),
#     Real(0, 5, name='CBFweight'),
#
#     Categorical([0.0], name="SSLIMweight"),#Real(0, 5, name="SSLIMweight"),
#     Real(0, 5, name='ALSweight'),
#     Categorical([0.0], name="LFMCFweight"),
#     Categorical([0.0], name="SLIMBPRweight"),#Real(0, 5, name='SLIMBPRweight'),
#
#     Categorical([0.0], name="SVDweight"),#Real(0, 5, name="SVDweight"),
#     Categorical([0.0], name="SVDweightSI"),  # Real(0, 5, name="SVDweightSI"),
#
#     Categorical([0.0], name="P3weight"),#Real(0, 5, name='P3weight'),
#     Real(0, 5, name='RP3weight'),  # Actually 0
#     Categorical([0.0], name="P3weightSI"),#Real(0, 5, name='P3weightSI'),
#     Real(0, 5, name='RP3weightSI'),
#
#     Categorical([True], name="normalize"),
#     Categorical([0.0], name="threshold"),#Integer(0, 5, name="threshold")
# ]

# names[HybridRecommenderWithTopK] = "HybridRecommenderWithTopK_tree_side_infoyesno_normalized_big_no_threshold"
# spaces[HybridRecommenderWithTopK] = [
#     Real(0.0001, 5, name='TopPopweight'),
#     Real(0, 5, name='IBCFweight'),
#     Real(0, 5, name='UBCFweight'),
#     Categorical([0.0], name="SSLIMweight"),#Real(0, 5, name='IBCFweightSI'),
#     Real(0, 5, name='UBCFweightSI'),
#     Real(0, 5, name='CBFweight'),
#
#     Categorical([0.0], name="SSLIMweight"),#Real(0, 5, name="SSLIMweight"),
#     Real(0, 5, name='ALSweight'),
#     Categorical([0.0], name="LFMCFweight"),
#     Real(0, 5, name='SLIMBPRweight'),
#
#     Categorical([0.0], name="SVDweight"),#Real(0, 5, name="SVDweight"),
#     Real(0, 5, name="SVDweightSI"),
#
#     Real(0, 5, name='P3weight'),
#     Real(0, 5, name='RP3weight'),
#     Real(0, 5, name='P3weightSI'),
#     Real(0, 5, name='RP3weightSI'),
#
#     Categorical([True], name="normalize"),
#     Categorical([0.0], name="threshold"),#Integer(0, 5, name="threshold")
# ]

# names[HybridRecommenderWithTopK] = "HybridRecommenderWithTopK_tree_side_infoyesno_normalized_mid_no_threshold"
# spaces[HybridRecommenderWithTopK] = [
#     Real(0.0001, 5, name='TopPopweight'),
#     Real(0, 5, name='IBCFweight'),
#     Real(0, 5, name='UBCFweight'),
#     Categorical([0.0], name="SSLIMweight"),#Real(0, 5, name='IBCFweightSI'),
#     Real(0, 5, name='UBCFweightSI'),
#     Real(0, 5, name='CBFweight'),
#
#     Categorical([0.0], name="SSLIMweight"),#Real(0, 5, name="SSLIMweight"),
#     Real(0, 5, name='ALSweight'),
#     Categorical([0.0], name="LFMCFweight"),
#     Real(0, 5, name='SLIMBPRweight'),
#
#     Categorical([0.0], name="SVDweight"),#Real(0, 5, name="SVDweight"),
#     Categorical([0.0], name="SVDweightSI"),#Real(0, 5, name="SVDweightSI"),
#
#     Categorical([0.0], name="P3weight"),#Real(0, 5, name='P3weight'),
#     Real(0, 5, name='RP3weight'),
#     Categorical([0.0], name="P3weightSI"),#Real(0, 5, name='P3weightSI'),
#     Real(0, 5, name='RP3weightSI'),
#
#     Categorical([True], name="normalize"),
#     Categorical([0.0], name="threshold"),#Integer(0, 5, name="threshold")
# ]

# names[HybridRecommenderWithTopK] = "HybridRecommenderWithTopK_exploit_previous_tree_side_infoyesno_normalized_no_threshold"
# spaces[HybridRecommenderWithTopK] = [
#     Real(0.0001, 1, name='TopPopweight'),
#     Real(0, 2, name='IBCFweight'),
#     Real(0, 2, name='UBCFweight'),
#     Categorical([0.0], name="SSLIMweight"),#Real(0, 5, name='IBCFweightSI'),
#     Real(0, 2, name='UBCFweightSI'),
#     Real(0, 1, name='CBFweight'),
#
#     Categorical([0.0], name="SSLIMweight"),#Real(0, 5, name="SSLIMweight"),
#     Real(4, 5, name='ALSweight'),
#     Categorical([0.0], name="LFMCFweight"),
#     Real(0, 2, name='SLIMBPRweight'),
#
#     Categorical([0.0], name="SVDweight"),#Real(0, 5, name="SVDweight"),
#     Categorical([0.0], name="SVDweightSI"),#Real(0, 5, name="SVDweightSI"),
#
#     Categorical([0.0], name="P3weight"),#Real(0, 5, name='P3weight'),
#     Real(0, 1, name='RP3weight'),
#     Categorical([0.0], name="P3weightSI"),#Real(0, 5, name='P3weightSI'),
#     Real(2, 4, name='RP3weightSI'),
#
#     Categorical([True], name="normalize"),
#     Categorical([0.0], name="threshold"),#Integer(0, 5, name="threshold")
# ]

names[HybridRecommenderWithTopK] = "HybridRecommenderWithTopK_exploit_previous_tree_side_infoyesno_normalized_no_threshold_smaller"
spaces[HybridRecommenderWithTopK] = [
    Real(0.0001, 1, name='TopPopweight'),
    Real(0, 2, name='IBCFweight'),
    Real(0, 2, name='UBCFweight'),
    Categorical([0.0], name="IBCFweightSI"),#Real(0, 5, name='IBCFweightSI'),
    Real(0, 2, name='UBCFweightSI'),
    Real(0, 1, name='CBFweight'),

    Categorical([0.0], name="SSLIMweight"),#Real(0, 5, name="SSLIMweight"),
    Real(4, 5, name='ALSweight'),
    Categorical([0.0], name="LFMCFweight"),
    Categorical([0.0], name='SLIMBPRweight'),

    Categorical([0.0], name="SVDweight"),#Real(0, 5, name="SVDweight"),
    Categorical([0.0], name="SVDweightSI"),#Real(0, 5, name="SVDweightSI"),

    Categorical([0.0], name="P3weight"),#Real(0, 5, name='P3weight'),
    Real(0, 1, name='RP3weight'),
    Categorical([0.0], name="P3weightSI"),#Real(0, 5, name='P3weightSI'),
    Real(2, 4, name='RP3weightSI'),

    Categorical([True], name="normalize"),
    Categorical([0.0], name="threshold"),#Integer(0, 5, name="threshold")
]