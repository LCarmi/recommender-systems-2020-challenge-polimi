#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Cesare Bernardis
"""

import numpy as np
import scipy.sparse as sp

from sklearn.preprocessing import normalize

from recommenders.recommender import Recommender
from utils.official.Recommender_utils import check_matrix, similarityMatrixTopK
from utils.official.IR_feature_weighting import apply_feature_weighting


import time, sys


class P3alphaRecommender(Recommender):

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True,
                 topK=590, alpha=0.285, min_rating=0, implicit=True, normalize_similarity=False,
                 feature_weighting="TF-IDF", K=1.04, B=0.96):

        super().__init__(URM, ICM,exclude_seen)
        self.W_sparse = None

        self.topK = topK
        self.alpha = alpha
        self.min_rating = min_rating
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity
        self.feature_weighting = feature_weighting
        self.K = K
        self.B = B


    def fit(self):

        self.URM = apply_feature_weighting(self.URM, self.feature_weighting, K=self.K, B=self.B)

        #
        # if X.dtype != np.float32:
        #     print("P3ALPHA fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.URM.data[self.URM.data < self.min_rating] = 0
            self.URM.eliminate_zeros()
            if self.implicit:
                self.URM.data = np.ones(self.URM.data.size, dtype=np.float32)

        #Pui is the row-normalized urm
        Pui = normalize(self.URM, norm='l1', axis=1)

        #Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)
        #ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0


        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = similarity_block[row_in_block, :]
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1


            if time.time() - start_time_printBatch > 60:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))


        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)


        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)

        self.W_sparse = check_matrix(self.W_sparse, format='csr')

        # Precompute URM
        self.predicted_URM = self.URM.dot(self.W_sparse)


class RP3betaRecommender(Recommender):

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True,
                 topK=950, alpha=0.298, beta=0.288, min_rating=0, implicit=True, normalize_similarity=False,
                 feature_weighting=None, K=1.2, B=0.75):

        super().__init__(URM, ICM, exclude_seen)
        self.W_sparse = None

        self.topK = topK
        self.alpha = alpha
        self.beta = beta
        self.min_rating = min_rating
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity
        self.feature_weighting = feature_weighting
        self.K = K
        self.B = B

    def fit(self):

        self.URM = apply_feature_weighting(self.URM, self.feature_weighting, K=self.K, B=self.B)

        # if X.dtype != np.float32:
        #     print("RP3beta fit: For memory usage reasons, we suggest to use np.float32 as dtype for the dataset")

        if self.min_rating > 0:
            self.URM.data[self.URM.data < self.min_rating] = 0
            self.URM.eliminate_zeros()
            if self.implicit:
                self.URM.data = np.ones(self.URM.data.size, dtype=np.float32)

        # Pui is the row-normalized urm
        Pui = normalize(self.URM, norm='l1', axis=1)

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        degree = np.zeros(self.URM.shape[1])

        nonZeroMask = X_bool_sum != 0.0

        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)

        # ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del (X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = np.multiply(similarity_block[row_in_block, :], degree)
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

            if time.time() - start_time_printBatch > 60:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(Pui.shape[1], Pui.shape[1]))

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)

        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)

        self.W_sparse = check_matrix(self.W_sparse, format='csr')

        # Precompute URM
        self.predicted_URM = self.URM.dot(self.W_sparse)

        print("RP3BetaRecommender training computed in {:.2f} seconds".format(time.time() - start_time_printBatch))


class P3alphaRecommenderSI(P3alphaRecommender):
    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True, topK=570, alpha=0.30, min_rating=0, implicit=True,
                 normalize_similarity=False, feature_weighting="BM25", K=1.04, B=0.96, omega=1):

        super().__init__(URM, ICM, exclude_seen, topK, alpha, min_rating, implicit, normalize_similarity,
                         feature_weighting, K, B)

        self.add_side_information(omega)


class RP3betaRecommenderSI(RP3betaRecommender):

    def __init__(self, URM: sp.csr_matrix, ICM, exclude_seen=True, topK=425, alpha=0.366, beta=0.683, min_rating=0,
                 implicit=True, normalize_similarity=False, feature_weighting=None, K=1.2, B=0.75, omega=39.15):

        super().__init__(URM, ICM, exclude_seen, topK, alpha, beta, min_rating, implicit, normalize_similarity,
                         feature_weighting, K, B)
        self.add_side_information(omega)

