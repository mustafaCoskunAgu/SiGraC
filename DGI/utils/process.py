import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import torch.nn as nn

def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret

# Process a (subset of) a TU dataset into standard form
def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))
       
    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks

def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))
    
    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adjRA(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    DA = d_mat_inv_sqrt.dot(adj);
    
    return adj.dot(DA).tocoo()
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_adjCN(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(adj).tocoo()

def normalize_adjAA(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(np.log(rowsum), -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    DA = d_mat_inv_sqrt.dot(adj);
    return adj.dot(DA).tocoo()

def normalize_adjSalton (adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    Dtemp = d_mat_inv_sqrt @ d_mat_inv_sqrt.T
    
    CNmat = adj @ adj
    result = CNmat @ Dtemp
    return result

def mymaximum (A, B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data <= 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)

def myminimum(A,B):
    BisBigger = A-B
    BisBigger.data = np.where(BisBigger.data >= 0, 1, 0)
    return A - A.multiply(BisBigger) + B.multiply(BisBigger)


def normalize_adjHDI(adj):
    adj = sp.coo_matrix(adj)

    rowsum = np.array(adj.sum(1))
    
    deg_row = np.tile(rowsum, (1,adj.shape[0]))
    
    #deg_row = deg_row.T
    deg_row = sp.coo_matrix(deg_row)
    
    sim = adj.dot(adj)
    
    #y = sim.copy().tocsr()
    #y.data.fill(1)
    X = sim.astype(bool).astype(int)
    deg_row = deg_row.multiply(X)
    
    deg_row = mymaximum(deg_row, deg_row.T)
    
    sim = sim/deg_row
    #sim = sp.coo_matrix(sim)
    whereAreNan = np.isnan(sim)
    whereAreInf = np.isinf(sim)
    sim[whereAreNan] = 0
    sim[whereAreInf] = 0
    
    sim = sp.coo_matrix(sim)
    #print(sim[0])
    return sim

def normalize_adjHPI(adj):
    adj = sp.coo_matrix(adj)

    rowsum = np.array(adj.sum(1))
    
    deg_row = np.tile(rowsum, (1,adj.shape[0]))
    
    #deg_row = deg_row.T
    deg_row = sp.coo_matrix(deg_row)
    
    sim = adj.dot(adj)
    
    #y = sim.copy().tocsr()
    #y.data.fill(1)
    X = sim.astype(bool).astype(int)
    deg_row = deg_row.multiply(X)
    
    deg_row = myminimum(deg_row, deg_row.T)
    
    sim = sim/deg_row
    #sim = sp.coo_matrix(sim)
    whereAreNan = np.isnan(sim)
    whereAreInf = np.isinf(sim)
    sim[whereAreNan] = 0
    sim[whereAreInf] = 0
    
    sim = sp.coo_matrix(sim)
    #print(sim[0])
    return sim

def normalize_adjJaccard(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    deg_row = np.tile(rowsum, (1,adj.shape[0]))
    deg_row = sp.coo_matrix(deg_row)
    
    sim = adj.dot(adj)
    X = sim.astype(bool).astype(int)
    deg_row = deg_row.multiply(X)
    deg_row = sp.triu(deg_row, k=0) + sp.triu(deg_row.T,k=0)

    sim = sim/(deg_row.multiply(X)-sim)
    whereAreNan = np.isnan(sim)
    whereAreInf = np.isinf(sim)
    sim[whereAreNan] = 0
    sim[whereAreInf] = 0
    
    sim = sp.coo_matrix(sim)
    return sim

def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr
def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())

def normalize_adjSorenson(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    sim = adj @ adj
    sim = sp.triu(sim, k=1)
    Dtemp = d_mat_inv_sqrt + d_mat_inv_sqrt.T
    
    Dtemp = sp.triu(Dtemp)
    
    return 2*sim.dot(Dtemp)

def linCCALap( H1, H2, outdim_size,adj,gamma):
        """
        An implementation of linear CCA
        # Arguments:
            H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
            outdim_size: specifies the number of new features
        # Returns
            A and B: the linear transformation matrices
            mean1 and mean2: the means of data for both views
        """
        L = normalize_adj(adj)
        r1 = 1e-4
        r2 = 1e-4

        m = H1.shape[0]
        o1 = H1.shape[1]
        o2 = H2.shape[1]
        
        m1 = np.mean(H1, axis=0)
        m2 = np.mean(H2, axis=0)
        H1bar = H1 - np.tile(m1, (m, 1))
        H2bar = H2 - np.tile(m2, (m, 1))

        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T,
                                                 H1bar) + r1 * np.identity(o1)
        SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T,
                                                 H2bar) + r2 * np.identity(o2)

        [D1, V1] = np.linalg.eigh(SigmaHat11)
        [D2, V2] = np.linalg.eigh(SigmaHat22)
        SigmaHat11RootInv = np.dot(
            np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = np.dot(
            np.dot(V2, np.diag(D2 ** -0.5)), V2.T)
        
        T1 = np.dot(np.dot(SigmaHat11RootInv,
                                   SigmaHat12), SigmaHat22RootInv)
        regulTerm = np.dot(np.dot(H1bar.T,
                                   L), H2bar)
        regulTerm = gamma*regulTerm
        T2 = np.dot(np.dot(SigmaHat11RootInv,
                                   regulTerm), SigmaHat22RootInv)

        # Tval = np.dot(np.dot(SigmaHat11RootInv,
        #                            SigmaHat12), SigmaHat22RootInv)
        Tval = T1-T2

        [U, D, V] = np.linalg.svd(Tval)
        V = V.T
        w1 = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
        w2 = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
        D = D[0:outdim_size]
        return w1,w2,m1,m2,D


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
