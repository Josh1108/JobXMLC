import torch
from typing import Dict
from scipy.sparse import csr_matrix
import xclib.evaluation.xc_metrics as xc_metrics
import time
from jobxmlc.core.data import DatasetGraphPrediction, GraphCollator, DatasetGraphPredictionEncode
from jobxmlc.core.network import HNSW
from jobxmlc.core.predict_main import encode_nodes
import os
import numpy as np
def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def remove_key_from_dict(_dict: Dict, _key: str = "name") -> Dict:
    """
    remove a key from a dictionary and return a new one
    :param _dict:
    :param _key:
    :return:
    """
    return {k: v for k, v in _dict.items() if k != _key}

def othermetrics(true_labels, pred_labels):
    true_lbls_t = true_labels.toarray()
    pred_lbls_t = pred_labels.toarray()
    kvals = [5,10,30,50,100]
    ndcg_k=[]
    rr = []
    y_true_list = true_lbls_t
    y_pred = pred_lbls_t
    for i in range(0,len(y_true_list)):
        y_t = y_true_list[i]
        y_t_len = len(y_t)
        y_p_index = np.flip(np.argsort(y_pred[i]),0)
        for i in range(0,len(y_p_index)):
            if y_p_index[i] in y_t:
                rr.append(1/float(i+1))
                break
        idcg = np.sum([1.0/np.log2(x+2) for x in range(0,y_t_len)])
        r = []
        ndcg = []
        for k in kvals:
            correct = len(np.intersect1d(y_t,y_p_index[0:k]))
            r.append(correct/float(y_t_len))
            dcg = 0
            for i in range(0,k):
                if y_p_index[i] in y_t:
                    dcg = dcg + 1.0/np.log2(i+2)
            ndcg.append(dcg/idcg)
        ndcg_k.append(ndcg)
    print(" mrr = "+str(np.mean(rr)))
    ndcg = np.mean(ndcg_k,axis=0)*100
    for i in range(0,len(kvals)):
        print(" ndcg@"+str(kvals[i])+"= "+str(ndcg[i]))
    return np.mean(rr), ndcg

# @nb.njit(cache=True)
def _recall(true_labels_indices, true_labels_indptr,
            pred_labels_data, pred_labels_indices, pred_labels_indptr, k,dir):
    skill_list =[]
    with open('../data/COLING/Y.txt','r') as f:
        lines = f.readlines()
        skill_list = [line.rstrip() for line in lines]
    print("skill list",skill_list[0])
        
    fracs = []
    for i in range(len(true_labels_indptr) - 1):
        _true_labels = true_labels_indices[true_labels_indptr[i]: true_labels_indptr[i + 1]]
        _data = pred_labels_data[pred_labels_indptr[i]: pred_labels_indptr[i + 1]]
        _indices = pred_labels_indices[pred_labels_indptr[i]: pred_labels_indptr[i + 1]]
        top_inds = np.argsort(_data)[::-1][:k]
        _pred_labels = _indices[top_inds]
        to_print_pred = [skill_list[x-1] for x in _pred_labels]
        to_print_true =[skill_list[x-1] for x in _true_labels]
        if k==20:
            with open(os.path.join(dir,'predictions_{}'.format(k)),'a') as f:
                f.write(str(to_print_pred))
                f.write('\n')
            with open(os.path.join(dir,'predictions_indexes_{}'.format(k)),'a') as f:
                f.write(str([x-1 for x in _pred_labels]))
                f.write('\n')
            with open(os.path.join(dir,'true_{}'.format(k)),'a') as f:
                f.write(str(to_print_true))
                f.write('\n')
            

        if(len(_true_labels) > 0):
            fracs.append(len(set(_pred_labels).intersection(
                set(_true_labels))) / len(_true_labels))
    return np.mean(np.array(fracs, dtype=np.float32))


def _precision(true_labels_indices, true_labels_indptr,
            pred_labels_data, pred_labels_indices, pred_labels_indptr, k):
    fracs =[]        
    for i in range(len(true_labels_indptr) - 1):
        _true_labels = true_labels_indices[true_labels_indptr[i]: true_labels_indptr[i + 1]]
        _data = pred_labels_data[pred_labels_indptr[i]: pred_labels_indptr[i + 1]]
        _indices = pred_labels_indices[pred_labels_indptr[i]: pred_labels_indptr[i + 1]]
        top_inds = np.argsort(_data)[::-1][:k]
        _pred_labels = _indices[top_inds]

    
        if(len(_pred_labels) > 0):
            fracs.append(len(set(_pred_labels).intersection(
                set(_true_labels))) / len(_pred_labels))
    return np.mean(np.array(fracs, dtype=np.float32))

def precision(true_labels,pred_labels,k,dir):
    return _precision(true_labels.indices.astype(np.int64), true_labels.indptr,
                   pred_labels.data, pred_labels.indices.astype(np.int64), pred_labels.indptr, k)

def recall(true_labels, pred_labels, k,dir):
    return _recall(true_labels.indices.astype(np.int64), true_labels.indptr,
                   pred_labels.data, pred_labels.indices.astype(np.int64), pred_labels.indptr, k,dir)

def remap_label_indices(trn_point_titles, label_titles):
    label_remapping = {}
    _new_label_index = len(trn_point_titles)
    trn_title_2_index = {x: i for i, x in enumerate(trn_point_titles)}

    for i, x in enumerate(label_titles):
        if(x in trn_title_2_index.keys()):
            label_remapping[i] = trn_title_2_index[x]
        else:
            label_remapping[i] = _new_label_index
            _new_label_index += 1

    print("_new_label_index =", _new_label_index)
    return label_remapping

def make_csr_from_ll(ll, num_z):
    data = []
    indptr = [0]
    indices = []
    for x in ll:
        indices += list(x)
        data += [1.0] * len(x)
        indptr.append(len(indices))

    return csr_matrix((data, indices, indptr), shape=(len(ll), num_z))

def create_params_dict(args, node_features, trn_X_Y,
                       graph, NUM_TRN_POINTS):
    DIM = node_features.shape[1]
    params = {**args}
    params.update(dict(hidden_dims=DIM,
                  feature_dim=DIM,
                  embed_dims=DIM,
                  lr=args.lr,
                  attention_lr=args.attention_lr
                  ))
    params["batch_size"] = args.batch_size
    params["reduction"] = "mean"
    params["batch_div"] = False
    params["num_epochs"] = args.num_epochs
    params["num_HN_epochs"] = args.num_HN_epochs
    params["dlr_factor"] = args.dlr_factor
    params["adjust_lr_epochs"] = set(
        [int(x) for x in args.adjust_lr.strip().split(",")])
    params["num_random_samples"] = args.num_random_samples
    params['encoder'] = args.encoder
    params["fanouts"] = [int(x.strip()) for x in args.fanouts.strip().split(
        ",") if len(x.strip()) != 0]
    params["num_labels"] = trn_X_Y.shape[1]
    params["graph"] = graph
    params["num_trn"] = NUM_TRN_POINTS
    params["inv_prop"] = xc_metrics.compute_inv_propesity(
        trn_X_Y, args.A, args.B)
    params["num_shortlist"] = args.num_shortlist
    params["num_HN_shortlist"] = args.num_HN_shortlist
    params["restrict_edges_num"] = args.restrict_edges_num
    params["restrict_edges_head_threshold"] = args.restrict_edges_head_threshold
    params["random_shuffle_nbrs"] = args.random_shuffle_nbrs
    return params


def create_validation_data(valid_tst_point_features, label_features, tst_X_Y_val,
                           params, TST_TAKE, head_net):
    """
    Create validation data. For val accuracy pattern observation
    This won't provide correct valdation picture as init(not graph) embeddings used and tst connection not added
    """
    if TST_TAKE == -1 :
        TST_TAKE = valid_tst_point_features.shape[0]

    if params['validation_freq'] != -1 and params['predict_ova'] == 0 :
        print("Creating shortlists for validation using base embeddings...")
        t1 = time.time()

        NGS = HNSW(
            M=100,
            efC=300,
            efS=params["num_shortlist"],
            num_threads=24)
        NGS.fit(label_features)

        prediction_shortlist, _ = NGS.predict(
            valid_tst_point_features[:TST_TAKE], params["num_shortlist"])
        print("prediction_shortlist.shape", prediction_shortlist.shape)
        print("Time taken in creating shortlists per point(ms)",
              ((time.time() - t1) / prediction_shortlist.shape[0]) * 1000)

    if params['validation_freq'] != -1:
        _start = params["num_trn"]
        _end = _start + TST_TAKE
        print("_start, _end = ", _start, _end)

        if params['predict_ova'] == 0 :
            val_dataset = DatasetGraphPrediction(
                _start, _end, prediction_shortlist)
        else:
            val_dataset = DatasetGraphPrediction(_start, _end, None)
        hcp = GraphCollator(head_net, params["num_labels"], None, train=0)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=512,
            num_workers=10,
            collate_fn=hcp,
            shuffle=False,
            pin_memory=False)

        val_data = dict(val_labels=tst_X_Y_val[:TST_TAKE, :],
                        val_loader=val_loader)
    else:
        val_data = None

    return val_data

def make_dir_path(dir):
    if not os.path.exists(dir):
        print("Making model dir...")
        os.makedirs(dir)


def sample_anns_nbrs(label_features, tst_point_features, num_nbrs=4):
    """
    Only works for case when a single graph can be built on all labels
    """
    BATCH_SIZE = 2000000

    t1 = time.time()
    print("building ANNS for neighbor sampling for NR scenario")
    label_NGS = HNSW(M=100, efC=300, efS=500, num_threads=24)
    label_NGS.fit(label_features)
    print("Done in ", time.time() - t1)
    t1 = time.time()

    tst_label_nbrs = np.zeros(
        (tst_point_features.shape[0], num_nbrs), dtype=np.int64)
    for i in range(0, tst_point_features.shape[0], BATCH_SIZE):
        print(i)
        _tst_label_nbrs, _ = label_NGS.predict(
            tst_point_features[i: i + BATCH_SIZE], num_nbrs)
        tst_label_nbrs[i: i + BATCH_SIZE] = _tst_label_nbrs

    print("Done in ", time.time() - t1)
    t1 = time.time()
    return tst_label_nbrs

def sample_hard_negatives(head_net, label_remapping, num_trn, params):
    label_nodes = [label_remapping[i] for i in range(len(label_remapping))]

    val_dataset = DatasetGraphPredictionEncode(label_nodes)
    hce = GraphCollator(head_net, params["num_labels"], None, train=0)
    encode_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=512,
        num_workers=4,
        collate_fn=hce,
        shuffle=False,
        pin_memory=True)

    label_embs_graph = np.zeros(
        (len(label_nodes),
         params["hidden_dims"]),
        dtype=np.float32)
    for batch in encode_loader:
        encoded = encode_nodes(head_net, batch)
        encoded = encoded.detach().cpu().numpy()
        label_embs_graph[batch["indices"]] = encoded

    val_dataset = DatasetGraphPredictionEncode(
        [i for i in range(num_trn)])
    hce = GraphCollator(head_net, params["num_labels"], None, train=0)
    encode_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=512,
        num_workers=4,
        collate_fn=hce,
        shuffle=False,
        pin_memory=True)

    trn_point_embs_graph = np.zeros(
        (num_trn, params["hidden_dims"]), dtype=np.float32)
    for batch in encode_loader:
        encoded = encode_nodes(head_net, batch)
        encoded = encoded.detach().cpu().numpy()
        trn_point_embs_graph[batch["indices"]] = encoded

    label_features = label_embs_graph
    trn_point_features = trn_point_embs_graph

    BATCH_SIZE = 2000000

        
    label_NGS = HNSW(
        M=100,
        efC=300,
        efS=params["num_HN_shortlist"],
        num_threads=24)
    label_NGS.fit(label_features)

    trn_label_nbrs = np.zeros(
        (trn_point_features.shape[0],
            params["num_HN_shortlist"]),
        dtype=np.int64)
    for i in range(0, trn_point_features.shape[0], BATCH_SIZE):
        _trn_label_nbrs, _ = label_NGS.predict(
            trn_point_features[i: i + BATCH_SIZE], params["num_HN_shortlist"])
        trn_label_nbrs[i: i + BATCH_SIZE] = _trn_label_nbrs
    prediction_shortlist_trn = trn_label_nbrs
    return prediction_shortlist_trn