# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import scipy.sparse.linalg as lg
import scipy.io as sio
#import hdf5storage as hd
from scipy.sparse.linalg import svds
import scipy.sparse as sp

__author__ = "Alan WANG"
__email__ = "alan1995wang@outlook.com"


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

    def learn_embedding(self):

        graph = self.g.G.to_undirected()
        A = nx.to_numpy_matrix(graph)
#        idSave={}
#        idSave['Net']=A
#        sio.savemat('Node2VecPPIAdj.mat',idSave)
        
        
#--------------------Open for RWR ---------------        
#        print("Page Rank")
#        norm_lap_mat = nx.laplacian_matrix(graph)
#        alpha = 0.1
#
#        M_g =  np.eye(graph.number_of_nodes())- alpha*norm_lap_mat
#        M_l = (1-alpha)*np.eye(graph.number_of_nodes())

#---------------------------Open this L3----------------------
#        print("L3G")
#        norm_lap_mat = nx.laplacian_matrix(graph)
#        mu = 0.1;
#        eta = 1e-6;
#        M_g  =  mu*norm_lap_mat + (1-mu)*np.eye(graph.number_of_nodes()) + eta*np.eye(graph.number_of_nodes())
#        M_l = np.eye(graph.number_of_nodes())
        
#----------------------Open for Katz--------------------------
#        print("Katz Measure")
#        self._beta = 0.0728
#        M_g = np.eye(graph.number_of_nodes()) - self._beta * A
#        M_l = self._beta * A
#-------------------------------------------------------------


#----------------------------Open this part for CN ---------------------
#       
        M_g = np.eye(graph.number_of_nodes())

        M_l = np.dot(A, A)
#        # -------------------------------------
        S = np.dot(np.linalg.inv(M_g), M_l)
        # s: \sigma_k
        u, s, vt = lg.svds(S, k=self._d // 2)
        sigma = np.diagflat(np.sqrt(s))
        X1 = np.dot(u, sigma)
        X2 = np.dot(vt.T, sigma)
        # self._X = X2
        self._X = np.concatenate((X1, X2), axis=1)
#--------------------LoadTopKEmbeddings--------------------------
#        print("Load Top-k Embedding")
#        mydata = sio.loadmat('TopKEmbedding50.mat')
#        self._X = mydata['Embedding']
#----------------------------------------------------------------
        
        
###################Correlation based S matrix--------------------
#        mat = hd.loadmat('S50.mat')
#        S = mat['S']
#        u, s, vt = lg.svds(S, k=self._d // 2)
#        sigma = np.diagflat(np.sqrt(s))
#        X1 = np.dot(u, sigma)
#        X2 = np.dot(vt.T, sigma)
#        # self._X = X2
#        self._X = np.concatenate((X1, X2), axis=1)
        
####################Direct SVD-------------------------------------
#        
#        print("LP3D SVD")
#        norm_lap_mat = nx.laplacian_matrix(graph)
#        mu = 0.9;
#        eta = 1e-6;
#        M_g  =  mu*norm_lap_mat + (1-mu)*np.eye(graph.number_of_nodes()) + eta*np.eye(graph.number_of_nodes())
#        #M_l = np.eye(graph.number_of_nodes())
#        U, Sigma, VT = svds(M_g, k=self._d)
#        Sigma = np.diag(Sigma)
#        W = np.matmul(U, np.sqrt(Sigma))
#        C = np.matmul(VT.T, np.sqrt(Sigma))
#    # print(np.sum(U))
#        embeddings = W + C
#        self._X = embeddings
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