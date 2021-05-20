# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:26:32 2020

@author: Secil
"""

# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import scipy.sparse.linalg as lg

__author__ = "Alan WANG"
__email__ = "alan1995wang@outlook.com"

import scipy.sparse as sp
class HOPE(object):
    def __init__(self, graph, d):
        '''
          d: representation vector dimension
        '''
        self._d = d
        self._graph = graph.G
        self.g = graph
        self._node_num = graph.node_size
        self.learn_embedding()
        
    def calc_A_hat(adj_matrix):
        nnodes = adj_matrix.shape[0]
        mu = 0.95
        eta = 1e-6
        A = adj_matrix# + sp.eye(nnodes)
        D_vec = np.sum(A, axis=1)
        D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
        D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
        return mu*D_invsqrt_corr @ A @ D_invsqrt_corr + (1-mu)*sp.eye(nnodes) + eta*sp.eye(nnodes)


    def learn_embedding(self):

        #graph = self.g.G
        graph = self.g.G.to_undirected()
        A = nx.to_numpy_matrix(graph)
        mu = 0.1;
        eta = 1e-6
        
        norm_lap_mat = nx.laplacian_matrix(graph)

        A =  mu*norm_lap_mat + (1-mu)*np.eye(graph.number_of_nodes()) + eta*np.eye(graph.number_of_nodes())
        #A = norm_lap_mat
        # self._beta = 0.0728

        # M_g = np.eye(graph.number_of_nodes()) - self._beta * A
        # M_l = self._beta * A
        print("dimension = ", self._d)
        print("PPR")
        M_g = np.eye(graph.number_of_nodes())
        M_l = np.dot(A, A)

        S = np.dot(np.linalg.inv(M_g), M_l)
        # s: \sigma_k
        u, s, vt = lg.svds(S, k=self._d // 2)
        sigma = np.diagflat(np.sqrt(s))
        X1 = np.dot(u, sigma)
        X2 = np.dot(vt.T, sigma)
        # self._X = X2
        self._X = np.concatenate((X1, X2), axis=1)

    @property
    def vectors(self):
        vectors = {}
        look_back = self.g.look_back_list
        for i, embedding in enumerate(self._X):
            vectors[look_back[i]] = embedding
        return vectors

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self._d))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
