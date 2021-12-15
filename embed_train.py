# -*- coding: utf-8 -*-

import ast
import logging
import os

#from gensim.models import Word2Vec
#from gensim.models.word2vec import LineSentence
from sklearn.preprocessing import scale
from GAE.train_model import gae_model
from OpenNE import gf, grarep, hope, lap, line, node2vec, sdne,RWR
from SVD.model import SVD_embedding
#from struc2vec import struc2vec
from utils import *
from scipy.linalg import fractional_matrix_power, inv
import numpy as np
import scipy.sparse as sp
#import hdf5storage as hd
import torch
import torch.nn as nn
import networkx as nx
import pandas as pd
from DGI.models import DGI, LogReg
from DGI.utils import process
from scipy.io import loadmat


#from utils import sparse_mx_to_torch_sparse_tensor
#from dataset import load


# Borrowed from https://github.com/PetarV-/DGI
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=True):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


# Borrowed from https://github.com/PetarV-/DGI
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)


# Borrowed from https://github.com/PetarV-/DGI
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4, s_bias1=None, s_bias2=None):
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = Readout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, diff, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)
        h_4 = self.gcn2(seq2, diff, sparse)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)

        return ret, h_1, h_2

    def embed(self, seq, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        h_2 = self.gcn2(seq, diff, sparse)
        return (h_1 + h_2).detach(), c.detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


def compute_pprAdj(adj, alpha=0.2, self_loop=True):
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    at = d_mat_inv_sqrt @ adj @  d_mat_inv_sqrt
    return alpha * inv((np.eye(adj.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1

def _scaleSimMat(A):
    """Scale rows of similarity matrix"""
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(np.float)/col[:, None]

    return A

def PPMI_matrix(M):
    """ Compute Positive Pointwise Mutual Information Matrix"""
    M = _scaleSimMat(M)
    n = M.shape[0]
    col = np.asarray(M.sum(axis=0), dtype=float)
    col = col.reshape((1, n))
    row = np.asarray(M.sum(axis=1), dtype=float)
    row = row.reshape((n, 1))
    D = np.sum(col)

    np.seterr(all='ignore')
    PPMI = np.log(np.divide(D*M, np.dot(row, col)))
    PPMI[np.isnan(PPMI)] = 0
    PPMI[PPMI < 0] = 0

def embedding_training(args, train_graph_filename):
    if args.method == 'struc2vec':
        g = read_for_struc2vec(train_graph_filename)
    elif args.method == 'GAE':
        if args.input == 'YeastAdj.mat':
            g = load_mat_data()
        else:
            g = read_for_gae(train_graph_filename)
    elif args.method == 'DGI':
        if args.input == 'YeastAdj.mat':
            g = load_mat_data()
        else:
            g = read_for_gae(train_graph_filename)
    elif args.method == 'SDGI':
        if args.input == 'YeastAdj.mat':
            g = load_mat_data()
        else:    
            g = read_for_gae(train_graph_filename)
    elif args.method == 'SVD':
        g = read_for_SVD(train_graph_filename, weighted=args.weighted)
    else:
        if args.input == 'YeastAdj.mat':
            g = read_for_OpenNE_from_mat(args.input)
        else:
            g = read_for_OpenNE(train_graph_filename, weighted=args.weighted)

    _embedding_training(args, G_=g)

    return


def load_mat_data():
    ne = loadmat('YeastAdj.mat')
    ne = ne['adj']
    G=nx.from_numpy_matrix(ne)
    node_list=list(G.nodes)
    adj = nx.adjacency_matrix(G, nodelist=node_list)
    print("Graph Loaded...")
    return (adj,node_list)
    

def _embedding_training(args, G_=None):
    seed=args.seed

    if args.method == 'struc2vec':
        logging.basicConfig(filename='./src/bionev/struc2vec/struc2vec.log', filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s')
        if (args.OPT3):
            until_layer = args.until_layer
        else:
            until_layer = None

        G = struc2vec.Graph(G_, args.workers, untilLayer=until_layer)

        if (args.OPT1):
            G.preprocess_neighbors_with_bfs_compact()
        else:
            G.preprocess_neighbors_with_bfs()

        if (args.OPT2):
            G.create_vectors()
            G.calc_distances(compactDegree=args.OPT1)
        else:
            G.calc_distances_all_vertices(compactDegree=args.OPT1)

        print('create distances network..')
        G.create_distances_network()
        print('begin random walk...')
        G.preprocess_parameters_random_walk()

        G.simulate_walks(args.number_walks, args.walk_length)
        print('walk finished..\nLearning embeddings...')
        walks = LineSentence('random_walks.txt')
        model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1,
                         workers=args.workers, seed=seed)
        os.remove("random_walks.txt")
        model.wv.save_word2vec_format(args.output)
    elif args.method == 'GAE':
        if args.input == 'STRING-EXP.mat':
            model = gae_model(args)
            G, node_list = load_mat_data()
            model.train(G)
            # save embeddings
            model.save_embeddings(args.output, node_list)
        else:
            
            model = gae_model(args)
            G = G_[0]
            node_list = G_[1]
            model.train(G)
            # save embeddings
            model.save_embeddings(args.output, node_list)
    elif args.method == 'SDGI':
        nb_epochs = 200
        patience = 20
        lr = 0.001
        l2_coef = 0.0
        hid_units = 100
        sparse = False
        verbose=True
        alpha = 0.2

        adj = G_[0]
        #diff = alpha * inv((np.eye(adj.shape[0]) - (1 - alpha) * (adj + sp.eye(adj.shape[0]))))
        #diff = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        #diff = diff.todense()
        
        
        #diff = compute_pprAdj(adj,alpha)
        node_list = G_[1]
        # datafile = 'expression_data.tsv'
        # normalize = True
        # df = pd.read_csv(datafile, sep='\t', header=0)
        # df.columns = [int(x[1:]) - 1 for x in df.columns]
        # if normalize==True:
        #     df = df[node_list]
        #     df = pd.DataFrame(scale(df, axis=0))
        # t_data = df.T
        # features = t_data.to_numpy()
        
        features = sp.identity(adj.shape[0])
        features, _ = process.preprocess_features(features)
        
        
        if args.embTech == 'DGI':
            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'CN':
            adj = process.normalize_adjCN(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'AA':
            adj = process.normalize_adjAA(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'Jaccard':
            adj = process.normalize_adjJaccard(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'RA':
            adj = process.normalize_adjRA(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'Adj-HDI':
            diff = process.normalize_adjHDI(adj + sp.eye(adj.shape[0]))
            diff = diff.todense()
            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            adj = adj.todense()
        elif args.embTech =='Adj-Adj':
            diff = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            diff = diff.todense()
            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            adj = adj.todense()  
        elif args.embTech == 'Salton-Salton':
            diff = process.normalize_adjSalton(adj + sp.eye(adj.shape[0]))
            diff = diff.todense()
            adj = process.normalize_adjSalton(adj + sp.eye(adj.shape[0]))
            adj = adj.todense()      
        elif args.embTech == 'HDI-RA':
            diff = process.normalize_adjHDI(adj + sp.eye(adj.shape[0]))
            diff = diff.todense()
            adj = process.normalize_adjRA(adj + sp.eye(adj.shape[0]))
            adj = adj.todense()
        elif args.embTech == 'HDI-Rwr':
            diff = compute_pprAdj(adj,alpha)
            #diff = process.normalize_adjHDI(adj + sp.eye(adj.shape[0]))
            #diff = diff.todense()
            adj = process.normalize_adjHDI(adj + sp.eye(adj.shape[0]))
            adj = adj.todense()
        elif args.embTech == 'HPI':
            adj = process.normalize_adjHPI(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'Sorenson':
            adj = process.normalize_adjSorenson(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'Salton':
            adj = process.normalize_adjSalton(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'Adj-Rwr':
            diff = compute_pprAdj(adj,alpha)
            #diff = process.normalize_adjHDI(adj + sp.eye(adj.shape[0]))
            #diff = diff.todense()
            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            adj = adj.todense()
        elif args.embTech == 'Adj-Salton':
            #diff = compute_pprAdj(adj,alpha)
            diff = process.normalize_adjSalton(adj + sp.eye(adj.shape[0]))
            diff = diff.todense()
            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            adj = adj.todense()
        elif args.embTech == 'Adj-RA':
            #diff = compute_pprAdj(adj,alpha)
            diff = process.normalize_adjRA(adj + sp.eye(adj.shape[0]))
            diff = diff.todense()
            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            adj = adj.todense()
        elif args.embTech == 'Salton-Rwr':
            diff = compute_pprAdj(adj,alpha)
            
            adj = process.normalize_adjSalton(adj + sp.eye(adj.shape[0]))
            adj = adj.todense()
        elif args.embTech == 'Adj-HPI':
            diff = process.normalize_adjHPI(adj + sp.eye(adj.shape[0]))
            diff = diff.todense()
            
            adj = process.normalize_adjSalton(adj + sp.eye(adj.shape[0]))
            adj = adj.todense()
        else:
            print("No such embedding technique \n We are calling default DGI", args.embTech)
            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        #adj = adj.todense()
        ft_size = features.shape[1]
        print("Size of features", ft_size)
        #features.tocsr()
        #nb_classes = np.unique(labels).shape[0]
        #sparse = True
        sample_size = 2000
        batch_size = 4
    
    
        lbl_1 = torch.ones(batch_size, sample_size * 2)
        lbl_2 = torch.zeros(batch_size, sample_size * 2)
        lbl = torch.cat((lbl_1, lbl_2), 1)
    
        model = Model(ft_size, hid_units)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    
    
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0
    
        for epoch in range(nb_epochs):
    
            idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
            ba, bd, bf = [], [], []
            for i in idx:
                ba.append(adj[i: i + sample_size, i: i + sample_size])
                bd.append(diff[i: i + sample_size, i: i + sample_size])
                bf.append(features[i: i + sample_size])
    
            ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
            bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
            bf = np.array(bf).reshape(batch_size, sample_size, ft_size)
    
            if sparse:
                ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
                bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
            else:
                ba = torch.FloatTensor(ba)
                bd = torch.FloatTensor(bd)
    
            bf = torch.FloatTensor(bf)
            idx = np.random.permutation(sample_size)
            shuf_fts = bf[:, idx, :]
    
            if torch.cuda.is_available():
                bf = bf.cuda()
                ba = ba.cuda()
                bd = bd.cuda()
                shuf_fts = shuf_fts.cuda()
    
            model.train()
            optimiser.zero_grad()
    
            logits, __, __ = model(bf, shuf_fts, ba, bd, sparse, None, None, None)
    
            loss = b_xent(logits, lbl)
    
            loss.backward()
            optimiser.step()
    
            if verbose:
                print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))
    
            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'model.pkl')
            else:
                cnt_wait += 1
    
            if cnt_wait == patience:
                if verbose:
                    print('Early stopping!')
                break
    
        if verbose:
            print('Loading {}th epoch'.format(best_t))
        model.load_state_dict(torch.load('model.pkl'))
    
        if sparse:
            adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
            diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))
    
        features = torch.FloatTensor(features[np.newaxis])
        adj = torch.FloatTensor(adj[np.newaxis])
        diff = torch.FloatTensor(diff[np.newaxis])
        #features = features.cuda()
        #adj = adj.cuda()
        #diff = diff.cuda()
    
        embeds, _ = model.embed(features, adj, diff, sparse, None)
        output = args.output
        TenToNum = embeds.numpy()
        newembeds = TenToNum[0]
            
        fout = open(output, 'w')
        fout.write("{} {}\n".format(newembeds.shape[0], newembeds.shape[1]))
        for idx in range(newembeds.shape[0]):
            fout.write("{} {}\n".format(node_list[idx], ' '.join([str(x) for x in newembeds[idx, :]])))
        fout.close()
        
    elif args.method == 'DGI':
                # training params for DGI
        batch_size = 1
        nb_epochs = args.epochs
        patience = 20
        lr = 0.001
        l2_coef = 0.0
        drop_prob = 0.0
        hid_units = 100
        sparse = True #Small datasets make it True
        nonlinearity = 'prelu' # special name to separate parameters
        adj = G_[0]
        node_list = G_[1]
        features = sp.identity(adj.shape[0])
        # datafile = 'expression_data.tsv'
        # normalize = True
        # df = pd.read_csv(datafile, sep='\t', header=0)
        # df.columns = [int(x[1:]) - 1 for x in df.columns]
        # if normalize==True:
        #     df = pd.DataFrame(scale(df, axis=0))
        # t_data = df.T
        # features = t_data.to_numpy()
        # features = features[[node_list],:]
        # #features = features.T
        # #features = sp.diags(pr)
        
        """ RWR features 3 steps """
        #features = myrwr(adj, 0.15,3)
        
        features, _ = process.preprocess_features(features)
        
        nb_nodes = features.shape[0]
        ft_size = features.shape[1]
        
        #matlabData = hd.loadmat('CTD_DDA_HDI.mat')
        #adj = matlabData['adj']
        
        
        """ For large file use implementation in Python"""
        #adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        #adj = process.calc_ppr_exact(adj, 0.15)
        #adj = myrwr(adj + sp.eye(adj.shape[0]), 0.15, 10)
        if args.embTech == 'DGI':
            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'CN':
            adj = process.normalize_adjCN(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'AA':
            adj = process.normalize_adjAA(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'Jaccard':
            adj = process.normalize_adjJaccard(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'RA':
            adj = process.normalize_adjRA(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'HDI':
            adj = process.normalize_adjHDI(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'HPI':
            adj = process.normalize_adjHPI(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'Sorenson':
            adj = process.normalize_adjSorenson(adj + sp.eye(adj.shape[0]))
        elif args.embTech == 'Salton':
            adj = process.normalize_adjSalton(adj + sp.eye(adj.shape[0]))
        else:
            print("No such embedding technique \n We are calling default DGI", args.embTech)
            adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
            
            
        if sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        else:
            adj = (adj + sp.eye(adj.shape[0])).todense()
        
        features = torch.FloatTensor(features[np.newaxis])
        if not sparse:
            adj = torch.FloatTensor(adj[np.newaxis])
        
        
        
        
        model = DGI(ft_size, hid_units, nonlinearity)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
        
        if torch.cuda.is_available():
            print('Using CUDA')
            model.cuda()
            features = features.cuda()
            if sparse:
                sp_adj = sp_adj.cuda()
            else:
                adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()
        
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0
        
        for epoch in range(nb_epochs):
            model.train()
            optimiser.zero_grad()
        
            idx = np.random.permutation(nb_nodes)
            shuf_fts = features[:, idx, :]
        
            lbl_1 = torch.ones(batch_size, nb_nodes)
            lbl_2 = torch.zeros(batch_size, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)
        
            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda()
                lbl = lbl.cuda()
            
            logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None) 
        
            loss = b_xent(logits, lbl)
        
            #print('Loss:', loss)
        
            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'best_dgi.pkl')
            else:
                cnt_wait += 1
        
            if cnt_wait == patience:
                print('Early stopping!')
                break
        
            loss.backward()
            optimiser.step()
        
        print('Loading {}th epoch'.format(best_t))
        model.load_state_dict(torch.load('best_dgi.pkl'))
        
        embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
        
        output = args.output
        TenToNum = embeds.numpy()
        newembeds = TenToNum[0]
            
        fout = open(output, 'w')
        fout.write("{} {}\n".format(newembeds.shape[0], newembeds.shape[1]))
        for idx in range(newembeds.shape[0]):
            fout.write("{} {}\n".format(node_list[idx], ' '.join([str(x) for x in newembeds[idx, :]])))
        fout.close()

    elif args.method == 'SVD':
        SVD_embedding(G_, args.output, size=args.dimensions)
    else:
        if args.method == 'Laplacian':
            model = lap.LaplacianEigenmaps(G_, rep_size=args.dimensions)
        elif args.method == 'RWR':
            model = RWR.RWR(G_, rep_size=100)

        elif args.method == 'GF':
            model = gf.GraphFactorization(G_, rep_size=args.dimensions,
                                          epoch=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay)

        elif args.method == 'HOPE':
            model = hope.HOPE(graph=G_, d=args.dimensions)

        elif args.method == 'GraRep':
            model = grarep.GraRep(graph=G_, Kstep=args.kstep, dim=args.dimensions)

        elif args.method == 'DeepWalk':
            model = node2vec.Node2vec(graph=G_, path_length=args.walk_length,
                                      num_paths=args.number_walks, dim=args.dimensions,
                                      workers=args.workers, window=args.window_size, dw=True)

        elif args.method == 'node2vec':
            model = node2vec.Node2vec(graph=G_, path_length=args.walk_length,
                                      num_paths=args.number_walks, dim=args.dimensions,
                                      workers=args.workers, p=args.p, q=args.q, window=args.window_size)

        elif args.method == 'LINE':
            model = line.LINE(G_, epoch=args.epochs,
                              rep_size=args.dimensions, order=args.order)

        elif args.method == 'SDNE':
            encoder_layer_list = ast.literal_eval(args.encoder_list)
            model = sdne.SDNE(G_, encoder_layer_list=encoder_layer_list,
                              alpha=args.alpha, beta=args.beta, nu1=args.nu1, nu2=args.nu2,
                              batch_size=args.bs, epoch=args.epochs, learning_rate=args.lr)
        else:
            raise ValueError(f'Invalid method: {args.method}')

        print("Saving embeddings...")
        model.save_embeddings(args.output)

    return
