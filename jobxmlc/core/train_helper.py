import numpy as np
from xclib.data import data_utils
from jobxmlc.core.utils import remap_label_indices, make_csr_from_ll,create_validation_data, sample_anns_nbrs
from jobxmlc.core.network import GalaXCBase
from jobxmlc.core.data import DatasetGraph, GraphCollator,Graph
from jobxmlc.core.predict_main import predict,update_predicted,update_predicted_shortlist,run_validation, validate
from scipy.sparse import vstack,lil_matrix
from scipy.spatial import distance
import xclib.evaluation.xc_metrics as xc_metrics
import torch
import time

def load_txt_file(file_path):
    return [line.strip() for line in open(file_path, "r").readlines()]

def data_loader(dataset_path,embedding_path):
    trn_point_titles = load_txt_file("{}/trn_X.txt".format(dataset_path))
    tst_point_titles = load_txt_file("{}/tst_X.txt".format(dataset_path))
    label_titles = load_txt_file("{}/Y.txt".format(dataset_path))

    trn_point_features = np.load(f"{embedding_path}/train_embeddings.npy",allow_pickle=True)
    tst_point_features = np.load(f"{embedding_path}/test_embeddings.npy",allow_pickle=True)
    label_features = np.load(f"{embedding_path}/label_embeddings.npy",allow_pickle=True)

    trn_X_Y = data_utils.read_sparse_file(
        f"{dataset_path}/trn_X_Y.txt",force_header =True,zero_based= False)

    tst_X_Y = data_utils.read_sparse_file(
        f"{dataset_path}/tst_X_Y.txt",force_header=True, zero_based = False)
    data_dict = {"trn_point_titles":trn_point_titles,"tst_point_titles":tst_point_titles,"label_titles":label_titles,"trn_point_features":trn_point_features,"tst_point_features":tst_point_features,"label_features":label_features,"trn_X_Y":trn_X_Y,"tst_X_Y":tst_X_Y}
    return data_dict


def trn_frst_prep(params,trn_X_Y, valid_tst_point_features, label_features, tst_X_Y_val, TST_TAKE, hard_negs):
    head_net = GalaXCBase(params["num_labels"], params["hidden_dims"], params['device_names'], params["feature_dim"], params["fanouts"], params["graph"], params["embed_dims"],params['encoder'])

    head_optimizer = torch.optim.Adam([{'params': [head_net.classifier.classifiers[0].attention_weights], 'lr': params["attention_lr"]},
                                      {"params": [param for name, param in head_net.named_parameters() if name != "classifier.classifiers.0.attention_weights"], "lr": params["lr"]}], lr=params["lr"])

    val_data = create_validation_data(valid_tst_point_features, label_features, tst_X_Y_val,
                                      params, TST_TAKE,head_net)

    head_criterion = torch.nn.BCEWithLogitsLoss(reduction=params["reduction"])

    head_train_dataset = DatasetGraph(trn_X_Y, hard_negs) # Right now hard_negs are empty


    hc = GraphCollator(
        head_net,
        params["num_labels"],
        params["num_random_samples"],
        num_hard_neg=0)


    head_train_loader = torch.utils.data.DataLoader(
        head_train_dataset,
        batch_size=params["batch_size"],
        num_workers=10,
        collate_fn=hc,
        shuffle=True,
        pin_memory=False
    )

    head_net.move_to_devices()
    inv_prop = xc_metrics.compute_inv_propesity(trn_X_Y, params['A'], params['B'])
    return head_net, head_train_loader, val_data, head_criterion, head_optimizer,inv_prop



def test(dir,params,head_net,RUN_TYPE,node_features,valid_tst_point_features,adjecency_lists,NUM_TRN_POINTS,label_remapping,
             label_features, tst_X_Y_val, tst_X_Y_trn):
    if RUN_TYPE == "NR":
        # introduce the tst points into the graph, assume all tst points known
        # at once. For larger graphs, doing ANNS on trn_points, labels work
        # equally well.
        tst_point_nbrs = sample_anns_nbrs(
            node_features,
            valid_tst_point_features,
            params['prediction_introduce_edges'])
        val_adj_list_trn = [list(x) for x in tst_point_nbrs]

        for i, l in enumerate(val_adj_list_trn):
            for x in l:
                adjecency_lists[i + NUM_TRN_POINTS].append(x)
        new_graph = Graph(
            node_features,
            adjecency_lists,
            params['random_shuffle_nbrs'])
        head_net.graph = new_graph

    t1 = time.time()

    recall_5 = validate(head_net, params, label_remapping,
             label_features, valid_tst_point_features, tst_X_Y_val, tst_X_Y_trn, True, 100,dir)
    
    print("Prediction time Per point(ms): ",
          ((time.time() - t1) / valid_tst_point_features.shape[0]) * 1000)
    return recall_5

def train(params,head_net,head_train_loader,head_criterion,head_optimizer,inv_prop,val_data,tst_X_Y_trn):
    
    if params['mpt'] == 1 :
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(params["num_epochs"]):

        epoch_train_start_time = time.time()
        head_net.train()
        torch.set_grad_enabled(True)

        mean_loss = 0
        for batch_idx, batch_data in enumerate(head_train_loader):
            t1 = time.time()
            head_net.zero_grad()
            batch_size = batch_data['batch_size']

            if params['mpt'] == 1:
                with torch.cuda.amp.autocast():
                    out_ans = head_net.forward(batch_data)
                    loss = head_criterion(
                        out_ans, batch_data["Y"].to(
                            out_ans.get_device()))
            elif params['mpt'] == 0 :
                out_ans = head_net.forward(batch_data)
                loss = head_criterion(
                    out_ans, batch_data["Y"].to(
                        out_ans.get_device()))

            if params["batch_div"]:
                loss = loss / batch_size
            mean_loss += loss.item() * batch_size

            if params['mpt'] == 1:
                scaler.scale(loss).backward()  # loss.backward()
                scaler.step(head_optimizer)  # head_optimizer3.step()
                scaler.update()
            elif params['mpt'] == 0:
                loss.backward()
                head_optimizer.step()
            del batch_data

        epoch_train_end_time = time.time()
        mean_loss /= len(head_train_loader.dataset)
        print(
            "Epoch: {}, loss: {}, time: {} sec".format(
                epoch,
                mean_loss,
                epoch_train_end_time -
                epoch_train_start_time))
        if epoch in params["adjust_lr_epochs"]:
            for param_group in head_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * params["dlr_factor"]

        if val_data is not None and ((epoch == 0) or (epoch % params['validation_freq'] == 0) or (epoch == params["num_epochs"] - 1)) :
            val_predicted_labels = lil_matrix(val_data["val_labels"].shape)

            t1 = time.time()
            with torch.set_grad_enabled(False):
                for batch_idx, batch_data in enumerate(val_data["val_loader"]):
                    val_preds, val_short = predict(head_net, batch_data)

                    if not(val_short is None) :
                        partition_length = val_short.shape[1]
                        update_predicted_shortlist((batch_data["inputs"]), val_preds,
                                                   val_predicted_labels, val_short, None, 10)
                    else:
                        update_predicted(batch_data["inputs"], torch.from_numpy(val_preds),
                                         val_predicted_labels, None, 10)

            print(
                "Per point(ms): ",
                ((time.time() - t1) / val_predicted_labels.shape[0]) * 1000)
            recall_lis,prec_lis,ndcg,mrr = run_validation(val_predicted_labels.tocsr(
            ), val_data["val_labels"], {}, tst_X_Y_trn, inv_prop)

            print("\nrecall_metrics\n",recall_lis,"\nprec_metrics\n",prec_lis,"\nndcg",ndcg,"\nmrr",mrr)

def prepare_test_data_PR(tst_X_Y, tst_point_features):
    """
    Prepare test data for Partial reveal task in prediciton. We reveal some data during training and some during testing.
    """
    tst_valid_inds = np.where(
            tst_X_Y.indptr[1:] - tst_X_Y.indptr[:-1] > 1)[0] # Number of test points having at least one label
    
    # in original dataset some points in tst have no labels
    print("data points with 0 labels, these are discarded:", np.sum(
        tst_X_Y.indptr[1:] - tst_X_Y.indptr[:-1] == 0))

    valid_tst_point_features = tst_point_features[tst_valid_inds]
    valid_tst_X_Y = tst_X_Y[tst_valid_inds, :]

    val_adj_list = [valid_tst_X_Y.indices[valid_tst_X_Y.indptr[i]
        : valid_tst_X_Y.indptr[i + 1]] for i in range(len(valid_tst_X_Y.indptr) - 1)]

    # split test data into two halves.
    val_adj_list_trn = [x[:(len(x) // 2)] for x in val_adj_list]
    val_adj_list_val = [x[(len(x) // 2):] for x in val_adj_list]
    return val_adj_list_trn, val_adj_list_val, valid_tst_point_features, tst_valid_inds

def prepare_data(data_dict,args):
    """
    We use run_type NR for skill prediction (none of the labels revealed at test time)
    
    inputs: (in data_dict)
        trn_X_Y: csr matrix of jd v labels
        tst_X_Y: csr matrix of jd v labels
        trn_point_features: train numpy embedding array 
        tst_point_features: test numpy embedding array
        labels_features: labels numpy embedding array
        trn_point_titles: train set jd  
        tst_point_titles: test set jd
        label_titles: all labels text 
    """
    trn_X_Y = data_dict["trn_X_Y"]
    tst_X_Y = data_dict["tst_X_Y"]
    trn_point_features = data_dict["trn_point_features"]
    tst_point_features = data_dict["tst_point_features"]
    label_features = data_dict["label_features"]
    trn_point_titles = data_dict["trn_point_titles"]
    tst_point_titles = data_dict["tst_point_titles"]
    label_titles = data_dict["label_titles"]
    
    if args['run_type'] == "PR" :

        val_adj_list_trn, val_adj_list_val, valid_tst_point_features, tst_valid_inds = prepare_test_data_PR(tst_X_Y, tst_point_features)
       
        # add half of the test data to training data

        adj_list = [trn_X_Y.indices[trn_X_Y.indptr[i]: trn_X_Y.indptr[i + 1]]
                    for i in range(len(trn_X_Y.indptr) - 1)] + val_adj_list_trn

        trn_point_titles = trn_point_titles + \
            [tst_point_titles[i] for i in tst_valid_inds]

        # labels need to be remapped--esentially the id is changing. assume training data start from 0 and labels start from len(trn_point_titles)
        # needs to be done for adjacency list and defining the nodes, as label and jd are i the same space
        label_remapping = remap_label_indices(trn_point_titles, label_titles)

        adj_list = [[label_remapping[x] for x in subl] for subl in adj_list]

        temp = {v: k for k, v in label_remapping.items() if v >=
                len(trn_point_titles)}

        new_label_indices = sorted(list(temp.keys()))

        _x = [temp[x] for x in new_label_indices]
        new_label_features = label_features[_x]
        # graph features
        node_features = np.vstack(
            [trn_point_features, valid_tst_point_features, new_label_features])


        # add connections only between trn and lbl, tst points are lone nodes
        # and thus are not included in convs
        adjecency_lists = [[] for i in range(node_features.shape[0])]
        for i, l in enumerate(adj_list):
            for x in l:
                adjecency_lists[i].append(x)
                adjecency_lists[x].append(i)

        tst_X_Y_val = make_csr_from_ll(val_adj_list_val, trn_X_Y.shape[1])
        tst_X_Y_trn = make_csr_from_ll(val_adj_list_trn, trn_X_Y.shape[1])

        trn_X_Y = vstack([trn_X_Y, tst_X_Y_trn])

        NUM_TRN_POINTS = trn_point_features.shape[0]

        # training is only over training points, but graph contains half of testing points, labels and all of training points in PR

    elif args['run_type'] == "NR":
        tst_X_Y_val = tst_X_Y
        tst_X_Y_trn = lil_matrix(tst_X_Y_val.shape).tocsr()  # Made an empty csr matrix of the same shape as jds v labels csr
        valid_tst_point_features = tst_point_features  # numpy array copied

        adj_list = [trn_X_Y.indices[trn_X_Y.indptr[i]: trn_X_Y.indptr[i + 1]] 
                    for i in range(len(trn_X_Y.indptr) - 1)]        # Adjecency list created. jd's -> labels

        trn_point_titles = trn_point_titles + tst_point_titles
        

        # all_point_titles = trn_point_titles + tst_point_titles + label_titles
        
        label_remapping = remap_label_indices(trn_point_titles, label_titles)

        adj_list = [[label_remapping[x] for x in subl] for subl in adj_list]

        temp = {v: k for k, v in label_remapping.items() if v >=
                len(trn_point_titles)}
        
        
        new_label_indices = sorted(list(temp.keys()))
        # print(new_label_indices)
        _x = [temp[x] for x in new_label_indices]
        new_label_features = label_features[_x]

        node_features = np.vstack(
            [trn_point_features, valid_tst_point_features, new_label_features])

        
        adjecency_lists = [[] for i in range(node_features.shape[0])] # Single list of lists for all nodes. labels -> JD and JD -> label.
        

        for i, l in enumerate(adj_list):
            for x in l:
                adjecency_lists[i].append(x)
                adjecency_lists[x].append(i)
        tst_valid_inds = np.arange(tst_X_Y_val.shape[0]) # indexes of test data. (2029)

        NUM_TRN_POINTS = trn_point_features.shape[0] # Size of training data
    

    if args['restrict_edges_num'] >= 3: # if number of neighbours are restricted take frequent labels (args.restrict_edges_head_threshold) in dataset and change adjecency lists to reflect
                                      # the neighbour pruning.

        head_labels = np.where(np.sum(trn_X_Y.astype(np.bool),axis=0)>args['restrict_edges_head_threshold'])[1]

        for lbl in head_labels:
            if lbl!=0:
                continue
            _nid = label_remapping[lbl]
            distances = distance.cdist([node_features[_nid]], [node_features[x] for x in adjecency_lists[_nid]], "cosine")[0]
            if args['take_most_frequent_labels']:
                sorted_indices = np.argsort(-distances)
            else:
                sorted_indices = np.argsort(-distances)

            new_nbrs = []
            for k in range(min(args['restrict_edges_num'], len(sorted_indices))):
                new_nbrs.append(adjecency_lists[_nid][sorted_indices[k]])
            adjecency_lists[_nid] = new_nbrs

    # Frequency part - uncomment to run frequency exp
    
    # n_a = np.sum(trn_X_Y.astype(np.bool),axis=0)
    # n_a = np.asarray(n_a).flatten()
    # #sorted_indices = np.argsort(-n_a) # Decreasing
    # sorted_indices = np.argsort(n_a)
    # #sorted_indices = np.asarray(n_a).flatten()

    # for j in range(trn_X_Y.shape[0]):
    #     new_nbrs=[]
    #     for i in range(len(sorted_indices)):
    #         _nid = label_remapping[sorted_indices[i]]
    #         if _nid in adjecency_lists[j]:
    #             new_nbrs.append(_nid)
    #     adjecency_lists[j] = new_nbrs
    # Printing number of edges
    # print(len(adjecency_lists))
    # counter =0
    # for item in adjecency_lists:
    #     counter+=  len(item)
    # print(counter)
    return tst_valid_inds, trn_X_Y, tst_X_Y_trn, tst_X_Y_val, node_features, valid_tst_point_features, label_remapping, adjecency_lists, NUM_TRN_POINTS