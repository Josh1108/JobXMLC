import torch
from typing import Dict
from scipy.sparse import csr_matrix
import xclib.evaluation.xc_metrics as xc_metrics
import time
from jobxmlc.core.data import DatasetGraphPrediction, GraphCollator
from jobxmlc.core.network import HNSW
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
                       graph, NUM_PARTITIONS, NUM_TRN_POINTS):
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