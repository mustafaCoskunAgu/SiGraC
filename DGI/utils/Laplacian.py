# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:31:11 2020

@author: Secil
"""

import numpy as np
import scipy.sparse as sp

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

A = [[0, 1, 2], 
    [1, 0, 4],
    [2, 4, 0]]

A2 = [[1, 1, 2], 
    [1, 1, 4],
    [2, 4, 1]]

L = normalize_adj(A)

LE = normalize_adj(A2)