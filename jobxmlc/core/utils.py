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
                  lr=args['lr'],
                  attention_lr=args['attention_lr']
                  ))
    params["reduction"] = "mean"
    params["batch_div"] = False
    params["adjust_lr_epochs"] = set(
        [int(x) for x in args['adjust_lr'].strip().split(",")])
    params["fanouts"] = [int(x.strip()) for x in args['fanouts'].strip().split(
        ",") if len(x.strip()) != 0]
    params["num_labels"] = trn_X_Y.shape[1]
    params["graph"] = graph
    params["num_trn"] = NUM_TRN_POINTS
    params["inv_prop"] = xc_metrics.compute_inv_propesity(
        trn_X_Y, args['A'], args['B'])
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

        NGS = HNSW(
            M=100,
            efC=300,
            efS=params["num_shortlist"],
            num_threads=24)
        NGS.fit(label_features)

        prediction_shortlist, _ = NGS.predict(
            valid_tst_point_features[:TST_TAKE], params["num_shortlist"])

    if params['validation_freq'] != -1:
        _start = params["num_trn"]
        _end = _start + TST_TAKE

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