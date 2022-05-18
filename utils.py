from collections import defaultdict as ddict
from os import path
from typing import List

import numpy as np
import torch
import torch.utils.data
import xclib.evaluation.xc_metrics as xc_metrics
from scipy.sparse import csr_matrix, lil_matrix, vstack
from scipy.spatial import distance
from tqdm import tqdm
import recsys_metrics
import predict_main
from data import *
from network import *


def remap_label_indices(trn_point_titles, label_titles):
    label_remapping = {}
    _new_label_index = len(trn_point_titles)
    trn_title_2_index = {x: i for i, x in enumerate(trn_point_titles)}

    for i, x in enumerate(label_titles):
        if x in trn_title_2_index.keys():
            label_remapping[i] = trn_title_2_index[x]
        else:
            label_remapping[i] = _new_label_index
            _new_label_index += 1

    print("_new_label_index =", _new_label_index)
    return label_remapping


def coorelation(label_remapping, adj_list, temp_lis, threshold=200) -> List[list]:
    """ sorted frequency of labels-labels
    :threshold: currently as 200
    :return: returns a list of list of labels which are kept at a threshold of 200
    """
    # print(len(label_remapping))
    # print(temp_lis)
    x = [0 for x in range(len(label_remapping) - 1)]
    llist = [x for i in range(len(label_remapping))]
    ret_llist = llist
    for i, job in enumerate(adj_list):
        temp = job
        for j, x in enumerate(temp):
            for k, skill in enumerate(temp):
                if x == skill:
                    continue
                llist[temp_lis[x]][temp_lis[skill]] += 1
    for num, l in enumerate(llist):
        ret_llist[num] = sorted(range(len(l)), key=lambda k: l[k])
    # Removing frequent skills
    # threshold set to 200
    dicti = ddict(int)
    for i, job in enumerate(adj_list):
        temp = job
        for j, x in enumerate(temp):
            dicti[x] += 1
    for num, l in enumerate(ret_llist):
        temp_l = []
        for item in l:
            if dicti[item] >= threshold:
                continue
            else:
                temp_l.append(item)
        ret_llist[num] = temp_l

    # end of removing frequent skills
    adj_list = [[label_remapping[x] for x in subl] for subl in ret_llist]
    return adj_list


def doc_doc_corr(adj_list, threshold: int = 10) -> List[list]:
    """"
    We need to add correlation between each doc. The way to do that is find an intersection in number of labels. For that, we find
    an intersection
    :args: threshold-> greater than equal to
        TODO:
    :args: threshold_common_labels -> number of common labels in documents ( threshold ~10)
    :args: threshold_common_head_labels -> number of common head labels in documents ( threshold ~10)
    :args: threshold_common_tail_labels ->number of common tail labels in documents ( threshold ~10)
    
    """
    basepath = path.dirname(__file__)
    dirpath = path.abspath(path.join(basepath, "..", "data", "COLING", "doc-doc"))
    file_list = os.listdir(dirpath)
    new_file = f'doc-doc-{threshold}.pkl'

    if new_file in file_list:
        with open(path.join(dirpath, new_file), 'rb') as fp:
            llist = pickle.load(fp)
        return llist

    else:
        llist = [adj_list[i] for i in range(len(adj_list))]
        for i, row in tqdm(enumerate(adj_list)):
            for j in range(i + 1, len(adj_list)):
                r = len(set(row) & set(adj_list[j]))
                if r >= threshold:
                    llist[i].append(j)
        with open(path.join(dirpath, new_file), 'wb') as fp:
            pickle.dump(llist, fp)
    return llist


def make_csr_from_ll(ll, num_z):
    data = []
    indptr = [0]
    indices = []
    for x in ll:
        indices += list(x)
        data += [1.0] * len(x)
        indptr.append(len(indices))

    return csr_matrix((data, indices, indptr), shape=(len(ll), num_z))


def mrr(true_labels, pred_labels):
    true_lbls_t = torch.from_numpy(true_labels.toarray())
    pred_lbls_t = torch.from_numpy(pred_labels.toarray())
    return recsys_metrics.mean_reciprocal_rank(true_lbls_t, pred_lbls_t).int()


def ndcg(true_labels, pred_labels, k):
    true_lbls_t = torch.from_numpy(true_labels.toarray())
    pred_lbls_t = torch.from_numpy(pred_labels.toarray())
    return recsys_metrics.normalized_dcg(true_lbls_t, pred_lbls_t, k=k).int()


# @nb.njit(cache=True)
def _recall(true_labels_indices, true_labels_indptr,
            pred_labels_data, pred_labels_indices, pred_labels_indptr, k, dir):
    skill_list = []
    with open('../data/COLING/Y.txt', 'r') as f:
        lines = f.readlines()
        skill_list = [line.rstrip() for line in lines]
    print("skill list", skill_list[0])

    fracs = []
    for i in range(len(true_labels_indptr) - 1):
        _true_labels = true_labels_indices[true_labels_indptr[i]: true_labels_indptr[i + 1]]
        _data = pred_labels_data[pred_labels_indptr[i]: pred_labels_indptr[i + 1]]
        _indices = pred_labels_indices[pred_labels_indptr[i]: pred_labels_indptr[i + 1]]
        top_inds = np.argsort(_data)[::-1][:k]
        _pred_labels = _indices[top_inds]
        to_print_pred = [skill_list[x - 1] for x in _pred_labels]
        to_print_true = [skill_list[x - 1] for x in _true_labels]
        if k == 20:
            with open(os.path.join(dir, 'predictions_{}'.format(k)), 'a') as f:
                f.write(str(to_print_pred))
                f.write('\n')
            with open(os.path.join(dir, 'predictions_indexes_{}'.format(k)), 'a') as f:
                f.write(str([x - 1 for x in _pred_labels]))
                f.write('\n')
            with open(os.path.join(dir, 'true_{}'.format(k)), 'a') as f:
                f.write(str(to_print_true))
                f.write('\n')

        if (len(_true_labels) > 0):
            fracs.append(len(set(_pred_labels).intersection(
                set(_true_labels))) / len(_true_labels))
    return np.mean(np.array(fracs, dtype=np.float32))


def _precision(true_labels_indices, true_labels_indptr,
               pred_labels_data, pred_labels_indices, pred_labels_indptr, k):
    fracs = []
    for i in range(len(true_labels_indptr) - 1):
        _true_labels = true_labels_indices[true_labels_indptr[i]: true_labels_indptr[i + 1]]
        _data = pred_labels_data[pred_labels_indptr[i]: pred_labels_indptr[i + 1]]
        _indices = pred_labels_indices[pred_labels_indptr[i]: pred_labels_indptr[i + 1]]
        top_inds = np.argsort(_data)[::-1][:k]
        _pred_labels = _indices[top_inds]

        if (len(_pred_labels) > 0):
            fracs.append(len(set(_pred_labels).intersection(
                set(_true_labels))) / len(_pred_labels))
    return np.mean(np.array(fracs, dtype=np.float32))


def precision(true_labels, pred_labels, k, dir):
    return _precision(true_labels.indices.astype(np.int64), true_labels.indptr,
                      pred_labels.data, pred_labels.indices.astype(np.int64), pred_labels.indptr, k)


def recall(true_labels, pred_labels, k, dir):
    return _recall(true_labels.indices.astype(np.int64), true_labels.indptr,
                   pred_labels.data, pred_labels.indices.astype(np.int64), pred_labels.indptr, k, dir)


def create_params_dict(args, node_features, trn_X_Y,
                       graph, NUM_PARTITIONS, NUM_TRN_POINTS):
    DIM = node_features.shape[1]
    params = dict(hidden_dims=DIM,
                  feature_dim=DIM,
                  embed_dims=DIM,
                  lr=args["lr"],
                  attention_lr=args["attention_lr"]
                  )
    params["batch_size"] = args["batch_size"]
    params["reduction"] = "mean"
    params["batch_div"] = False
    params["num_epochs"] = args["num_epochs"]
    params["num_HN_epochs"] = args["num_HN_epochs"]
    params["dlr_factor"] = args["dlr_factor"]
    params["adjust_lr_epochs"] = set(
        [int(x) for x in args["adjust_lr"].strip().split(",")])
    params["num_random_samples"] = args["num_random_samples"]
    params["devices"] = [x.strip()
                         for x in args["devices"].strip().split(",") if len(x.strip()) != 0]

    params["fanouts"] = [int(x.strip()) for x in args["fanouts"].strip().split(
        ",") if len(x.strip()) != 0]
    params["num_partitions"] = NUM_PARTITIONS
    params["num_labels"] = trn_X_Y.shape[1]
    params["graph"] = graph
    params["num_trn"] = NUM_TRN_POINTS
    params["inv_prop"] = xc_metrics.compute_inv_propesity(
        trn_X_Y, args["A"], args["B"])
    params["num_shortlist"] = args["num_shortlist"]
    params["num_HN_shortlist"] = args["num_HN_shortlist"]
    params["restrict_edges_num"] = args["restrict_edges_num"]
    params["restrict_edges_head_threshold"] = args["restrict_edges_head_threshold"]
    params["random_shuffle_nbrs"] = args["random_shuffle_nbrs"]

    return params


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


def prepare_data(trn_X_Y, tst_X_Y, trn_point_features, tst_point_features, label_features,
                 trn_point_titles, tst_point_titles, label_titles, args, logging):
    """
    We use run_type NR for skill prediction ( none of the labels revealed at test time)
    
    inputs:
        trn_X_Y: csr matrix of jd v labels
        tst_X_Y: csr matrix of jd v labels
        trn_point_features: train numpy embedding array 
        tst_point_features: test numpy embedding array
        labels_features: labels numpy embedding array
        trn_point_titles: train set jd  
        tst_point_titles: test set jd
        label_titles: all labels text 
    """

    if (args["run_type"] == "PR"):
        tst_valid_inds = np.where(
            tst_X_Y.indptr[1:] - tst_X_Y.indptr[:-1] > 1)[0]  # indices of test points having at least one label

        # in original dataset some points in tst have no labels
        print("point with 0 labels:", np.sum(
            tst_X_Y.indptr[1:] - tst_X_Y.indptr[:-1] == 0))

        # selecting valid test points
        valid_tst_point_features = tst_point_features[tst_valid_inds]
        valid_tst_X_Y = tst_X_Y[tst_valid_inds, :]

        # job -> label adj list
        val_adj_list = [valid_tst_X_Y.indices[valid_tst_X_Y.indptr[i]
                                              : valid_tst_X_Y.indptr[i + 1]] for i in
                        range(len(valid_tst_X_Y.indptr) - 1)]

        # splitting adj_list for train/val -> if we have 20 labels, 10 go in trn, 10 in val.
        # This split is done sort of randomly ( first 10 taken for val).
        # This split could be a reason for low performance for PR method.

        val_adj_list_trn = [x[:(len(x) // 2)] for x in val_adj_list]
        val_adj_list_val = [x[(len(x) // 2):] for x in val_adj_list]

        # Adding the first half to train dataset - Partial reveal. Check: Is this part of training? That would be wrong.
        # What we expected was that test data would be connected to labels at testing time.

        adj_list = [trn_X_Y.indices[trn_X_Y.indptr[i]: trn_X_Y.indptr[i + 1]]
                    for i in range(len(trn_X_Y.indptr) - 1)] + val_adj_list_trn  # added test data to adj_list
        print("adjacency list", adj_list)
        print("training pointers", trn_X_Y.indices[trn_X_Y.indptr[0]: trn_X_Y.indptr[0 + 1]])
        trn_point_titles = trn_point_titles + \
                           [tst_point_titles[i] for i in tst_valid_inds]

        label_remapping = remap_label_indices(trn_point_titles, label_titles)
        adj_list = [[label_remapping[x] for x in subl] for subl in adj_list]

        temp = {v: k for k, v in label_remapping.items() if v >=
                len(trn_point_titles)}
        print("len(label_remapping), len(temp), len(trn_point_titles)",
              len(label_remapping), len(temp), len(trn_point_titles))

        new_label_indices = sorted(list(temp.keys()))

        _x = [temp[x] for x in new_label_indices]
        new_label_features = label_features[_x]
        lengths = [
            trn_point_features.shape,
            valid_tst_point_features.shape,
            new_label_features.shape]
        print("lengths, sum([x[0] for x in lengths])",
              lengths, sum([x[0] for x in lengths]))

        node_features = np.vstack(
            [trn_point_features, valid_tst_point_features, new_label_features])
        print("node_features.shape", node_features.shape)

        # add connections only between trn and lbl, tst points are lone nodes
        # and thus are not included in convs
        adjecency_lists = [[] for i in range(node_features.shape[0])]
        for i, l in enumerate(adj_list):
            for x in l:
                adjecency_lists[i].append(x)
                adjecency_lists[x].append(i)
        # adjacency_lists contain PR data as well.

        tst_X_Y_val = make_csr_from_ll(val_adj_list_val, trn_X_Y.shape[1])
        tst_X_Y_trn = make_csr_from_ll(val_adj_list_trn, trn_X_Y.shape[1])

        trn_X_Y = vstack([trn_X_Y, tst_X_Y_trn])

        NUM_TRN_POINTS = trn_point_features.shape[0]  # NUM_TRN_POINTS = true train points, rest are test data added.

    elif (args["run_type"] == "NR"):
        tst_X_Y_val = tst_X_Y
        tst_X_Y_trn = lil_matrix(
            tst_X_Y_val.shape).tocsr()  # Made an empty csr matrix of the same shape as jds v labels csr
        valid_tst_point_features = tst_point_features  # numpy array copied

        adj_list = [trn_X_Y.indices[trn_X_Y.indptr[i]: trn_X_Y.indptr[i + 1]]
                    for i in range(len(trn_X_Y.indptr) - 1)]  # Adjecency list created. jd's -> labels

        trn_point_titles = trn_point_titles + tst_point_titles

        # all_point_titles = trn_point_titles + tst_point_titles + label_titles

        label_remapping = remap_label_indices(trn_point_titles, label_titles)

        adj_list = [[label_remapping[x] for x in subl] for subl in adj_list]

        temp = {v: k for k, v in label_remapping.items() if v >=
                len(trn_point_titles)}
        # print(len(temp))

        logging.info("len(label_remapping), len(temp), len(trn_point_titles)" +
                     str(len(label_remapping)) + str(len(temp)) + str(len(trn_point_titles)))

        new_label_indices = sorted(list(temp.keys()))
        # print(new_label_indices)
        _x = [temp[x] for x in new_label_indices]
        new_label_features = label_features[_x]
        lengths = [
            trn_point_features.shape,
            valid_tst_point_features.shape,
            new_label_features.shape]

        node_features = np.vstack(
            [trn_point_features, valid_tst_point_features, new_label_features])
        logging.info("node_features.shape" + str(node_features.shape))

        adjecency_lists = [[] for i in range(
            node_features.shape[0])]  # Single list of lists for all nodes. labels -> JD and JD -> label.

        # if args["doc_doc:
        # print("===================Performing doc-doc-graph construction=======================")
        # adj_list = doc_doc_corr(adj_list) 

        for i, l in enumerate(adj_list):
            for x in l:
                adjecency_lists[i].append(x)
                adjecency_lists[x].append(i)
        # print("====================Performing label-label-matrix construction==============================")
        # llist = coorelation(label_remapping,adj_list,temp)
        # for i,a in enumerate(llist):
        #     for b in a:
        #         adjecency_lists[label_remapping[i]].append(b)

        # Here we have the adjacency list and it's shape is (20817,x)
        tst_valid_inds = np.arange(tst_X_Y_val.shape[0])  # indexes of test data. (2029)

        NUM_TRN_POINTS = trn_point_features.shape[0]  # Size of training data

    # uncomment the restrict edges part

    if (
            args["restrict_edges_num"] >= 3):  # if number of neighbours are restricted take frequent labels (
        # args["restrict_edges_head_threshold) in dataset and change adjecency lists to reflect
        # the neighbour pruning.

        head_labels = np.where(np.sum(trn_X_Y.astype(np.bool), axis=0) > args["restrict_edges_head_threshold"])[1]
        # print(trn_X_Y.astype(np.bool))
        # print("Value",np.sum( trn_X_Y.astype(np.bool),axis=0),np.sum( trn_X_Y.astype(np.bool),axis=0).shape)
        # print(np.where(np.sum( trn_X_Y.astype(np.bool),axis=0)==0))
        # print(labels,len(labels))
        logging.info(
            "Restricting edges: Number of head labels = {}".format(
                len(head_labels)))
        # print(head_labels)

        for lbl in head_labels:
            if lbl != 0:
                continue
            _nid = label_remapping[lbl]
            print("printing node id", _nid)
            distances = \
                distance.cdist([node_features[_nid]], [node_features[x] for x in adjecency_lists[_nid]], "cosine")[0]
            sorted_indices = np.argsort(-distances)

            new_nbrs = []
            for k in range(min(args["restrict_edges_num"], len(sorted_indices))):
                new_nbrs.append(adjecency_lists[_nid][sorted_indices[k]])
            adjecency_lists[_nid] = new_nbrs

    # Frequency part - uncomment to run frequency exp

    n_a = np.sum(trn_X_Y.astype(np.bool), axis=0)
    n_a = np.asarray(n_a).flatten()
    sorted_indices = np.argsort(-n_a)  # Decreasing
    # sorted_indices = np.argsort(n_a) # increasing
    # sorted_indices = np.asarray(n_a).flatten()

    for j in range(trn_X_Y.shape[0]):
        new_nbrs = []
        for i in range(len(sorted_indices)):
            _nid = label_remapping[sorted_indices[i]]
            if _nid in adjecency_lists[j]:
                new_nbrs.append(_nid)
        adjecency_lists[j] = new_nbrs

    return tst_valid_inds, trn_X_Y, tst_X_Y_trn, tst_X_Y_val, node_features, valid_tst_point_features, label_remapping, adjecency_lists, NUM_TRN_POINTS


def create_validation_data(valid_tst_point_features, label_features, tst_X_Y_val,
                           args, params, TST_TAKE, head_net):
    """
    Create validation data. For val accuracy pattern observation
    This won't provide correct valdation picture as init(not graph) embeddings used and tst connection not added
    TST_TAKE: num_validation, number of points to take for validation
    validation_freq: after how many epochs
    predict ova: predict out of vocab
    """
    if (TST_TAKE == -1):
        TST_TAKE = valid_tst_point_features.shape[0]  # take all of test.

    if (args["validation_freq"] != -1 and args["predict_ova"] == 0):
        print("Creating shortlists for validation using base embeddings...")
        prediction_shortlists = []
        t1 = time.time()
        NGS = HNSW(
            M=100,
            efC=300,
            efS=params["num_shortlist"],
            num_threads=24)
        NGS.fit(label_features)

        prediction_shortlist, _ = NGS.predict(
            valid_tst_point_features[:TST_TAKE], min(params["num_shortlist"], len(label_features)))
        prediction_shortlists.append(prediction_shortlist)
        prediction_shortlist = prediction_shortlists[0]
        del (prediction_shortlists)
        print("prediction_shortlist.shape", prediction_shortlist.shape)
        print("Time taken in creating shortlists per point(ms)",
              ((time.time() - t1) / prediction_shortlist.shape[0]) * 1000)

    if (args["validation_freq"] != -1):
        _start = params["num_trn"]
        _end = _start + TST_TAKE
        print("_start, _end = ", _start, _end)

        if (args["predict_ova"] == 0):
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
        encoded = predict_main.encode_nodes(head_net, batch)
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
        encoded = predict_main.encode_nodes(head_net, batch)
        encoded = encoded.detach().cpu().numpy()
        trn_point_embs_graph[batch["indices"]] = encoded

    label_features = label_embs_graph
    trn_point_features = trn_point_embs_graph

    prediction_shortlists_trn = []
    BATCH_SIZE = 2000000

    t1 = time.time()

    label_NGS = HNSW(
        M=100,
        efC=300,
        efS=params["num_HN_shortlist"],
        num_threads=24)
    label_NGS.fit(
        label_features)
    print("Done in ", time.time() - t1)
    t1 = time.time()

    trn_label_nbrs = np.zeros(
        (trn_point_features.shape[0],
         params["num_HN_shortlist"]),
        dtype=np.int64)
    for i in range(0, trn_point_features.shape[0], BATCH_SIZE):
        print(i)
        _trn_label_nbrs, _ = label_NGS.predict(
            trn_point_features[i: i + BATCH_SIZE], params["num_HN_shortlist"])
        trn_label_nbrs[i: i + BATCH_SIZE] = _trn_label_nbrs

    prediction_shortlists_trn.append(trn_label_nbrs)
    print("Done in ", time.time() - t1)
    t1 = time.time()

    prediction_shortlist_trn = prediction_shortlists_trn[0]
    return prediction_shortlist_trn
