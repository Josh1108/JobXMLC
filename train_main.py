from __future__ import print_function
from __future__ import division

import optuna.exceptions
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.utils.data

import numpy as np
import math
import time
import os
import pickle
import random
import nmslib
import sys
import argparse
import warnings
import logging
import json
from scipy.spatial import distance
from scipy.sparse import csr_matrix, lil_matrix, load_npz, hstack, vstack, save_npz
import wandb
from xclib.data import data_utils
from xclib.utils.sparse import normalize
import xclib.evaluation.xc_metrics as xc_metrics
from predict_main import predict, update_predicted, update_predicted_shortlist, run_validation, validate
from collections import defaultdict, Counter
from utils import prepare_data, create_params_dict, create_validation_data, sample_hard_negatives,sample_anns_nbrs
from data import Graph, DatasetGraph, GraphCollator
from network import JobXMLCBase
torch.manual_seed(22)
torch.cuda.manual_seed_all(22)
np.random.seed(22)


def test(dir, args, params, head_net, RUN_TYPE, node_features, valid_tst_point_features, adjecency_lists,
         NUM_TRN_POINTS, label_remapping,
         label_features, tst_X_Y_val, tst_exact_remove, tst_X_Y_trn):
    if (RUN_TYPE == "NR"):
        # introduce the tst points into the graph, assume all tst points known
        # at once. For larger graphs, doing ANNS on trn_points, labels work
        # equally well.
        tst_point_nbrs = sample_anns_nbrs(
            node_features,
            valid_tst_point_features,
            args["prediction_introduce_edges"])
        val_adj_list_trn = [list(x) for x in tst_point_nbrs]

        for i, l in enumerate(val_adj_list_trn):
            for x in l:
                adjecency_lists[i + NUM_TRN_POINTS].append(x)
        new_graph = Graph(
            node_features,
            adjecency_lists,
            args["random_shuffle_nbrs"])
        head_net.graph = new_graph

    t1 = time.time()
    recall_5 = validate(head_net, params, label_remapping,
                        label_features, valid_tst_point_features, tst_X_Y_val, tst_exact_remove, tst_X_Y_trn, True, 100,
                        dir)
    print("Prediction time Per point(ms): ",
          ((time.time() - t1) / valid_tst_point_features.shape[0]) * 1000)
    return recall_5


def train(args, params, head_net, head_train_loader, val_data, head_criterion, head_optimizer,logger,trial, tst_X_Y_trn, inv_prop):
    if (args["mpt"] == 1):
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(params["num_epochs"]):

        epoch_train_start_time = time.time()
        head_net.train()
        torch.set_grad_enabled(True)

        num_batches = len(head_train_loader.dataset) // params["batch_size"]
        mean_loss = 0
        for batch_idx, batch_data in enumerate(head_train_loader):
            t1 = time.time()
            head_net.zero_grad()
            batch_size = batch_data['batch_size']

            if (args["mpt"] == 1):
                with torch.cuda.amp.autocast():
                    out_ans = head_net.forward(batch_data)
                    loss = head_criterion(
                        out_ans, batch_data["Y"].to(
                            out_ans.get_device()))
            elif (args["mpt"] == 0):
                out_ans = head_net.forward(batch_data)
                loss = head_criterion(
                    out_ans, batch_data["Y"].to(
                        out_ans.get_device()))

            if params["batch_div"]:
                loss = loss / batch_size
            mean_loss += loss.item() * batch_size

            if (args["mpt"] == 1):
                scaler.scale(loss).backward()  # loss.backward()
                scaler.step(head_optimizer)  # head_optimizer3.step()
                scaler.update()
            elif (args["mpt"] == 0):
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
        logger.info(
            "Epoch: {}, loss: {}, time: {} sec".format(
                epoch,
                mean_loss,
                epoch_train_end_time -
                epoch_train_start_time))

        if (epoch in params["adjust_lr_epochs"]):
            for param_group in head_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * params["dlr_factor"]

        if (val_data is not None and (
                (epoch == 0) or (epoch % args["validation_freq"] == 0) or (epoch == params["num_epochs"] - 1))):
            val_predicted_labels = lil_matrix(val_data["val_labels"].shape)

            t1 = time.time()
            with torch.set_grad_enabled(False):
                for batch_idx, batch_data in enumerate(val_data["val_loader"]):
                    val_preds, val_short = predict(head_net, batch_data)

                    if (not (val_short is None)):
                        update_predicted_shortlist((batch_data["inputs"]), val_preds,
                                                   val_predicted_labels, val_short, None, 20)
                    else:
                        update_predicted(batch_data["inputs"], torch.from_numpy(val_preds),
                                         val_predicted_labels, None, 20)

            logger.info(
                "Per point(ms): ",
                ((time.time() - t1) / val_predicted_labels.shape[0]) * 1000)

            all_metrics = run_validation(val_predicted_labels.tocsr(
            ), val_data["val_labels"], tst_X_Y_trn, inv_prop, dir)
            print("acc = {}".format(all_metrics))
            logger.info("Recall:" + " ".join(all_metrics[0]) + " \nPrecision" + " ".join(all_metrics[1]))
        trial.report(all_metrics[0][0],epoch)
        wandb.log(data={"recall@5":all_metrics[0][0],
                        "recall@10":all_metrics[0][1],
                        "recall@20":all_metrics[0][2],
                        "recall@30":all_metrics[0][3],
                        "recall@50":all_metrics[0][4],
                        "recall@100":all_metrics[0][5],
                        "precision@5":all_metrics[1][0],
                        "precision@10":all_metrics[1][1],
                        "precision@20":all_metrics[1][2],
                        "precision@30":all_metrics[1][3],
                        "precision@50":all_metrics[1][4],
                        "precision@100":all_metrics[1][5],
                        "ndcg@5": all_metrics[2][0],
                        "ndcg@10": all_metrics[2][1],
                        "ndcg@20": all_metrics[2][2],
                        "ndcg@30": all_metrics[2][3],
                        "ndcg@50": all_metrics[2][4],
                        "ndcg@100": all_metrics[2][5],
                        "mrr":all_metrics[3]
                        },step=epoch)
        if trial.should_prune():
            wandb.run.summary["state"] = "pruned"
            wandb.finish(quiet=True)
            raise optuna.exceptions.TrialPruned()

def train_optuna(model_dir, args,trial):

    DATASET = args["dataset"]
    RUN_TYPE = args["run_type"]

    # ================LOGGING DETAILS ==========================

    EMB_TYPE = args["embedding_type"]
    TST_TAKE = args["num_validation"]
    NUM_TRN_POINTS = -1

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        filename="{}/models/GraphXMLBERT_log_{}_{}.txt".format(DATASET, RUN_TYPE, args["name"]),
                        level=logging.INFO)
    logger = logging.getLogger("main_logger")

    logger.info("================= STARTING NEW RUN =====================")

    logger.info(" ARGUMENTS ")
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    # ===========================   Data load   ===========================

    trn_point_titles = [
        line.strip() for line in open(
            "{}/trn_X.txt".format(DATASET),
            "r",
            encoding="latin").readlines()]
    tst_point_titles = [
        line.strip() for line in open(
            "{}/tst_X.txt".format(DATASET),
            "r",
            encoding="latin").readlines()]
    label_titles = [
        line.strip() for line in open(
            "{}/Y.txt".format(DATASET),
            "r",
            encoding="latin").readlines()]
    print("len(trn_point_titles), len(tst_point_titles), len(label_titles) = ", len(
        trn_point_titles), len(tst_point_titles), len(label_titles))
    trn_point_features = np.load(
        "{}/{}CondensedData/trn_point_embs.npy".format(DATASET, EMB_TYPE), allow_pickle=True)
    label_features = np.load(
        "{}/{}CondensedData/label_embs.npy".format(DATASET, EMB_TYPE), allow_pickle=True)

    tst_point_features = np.load(
        "{}/{}CondensedData/tst_point_embs.npy".format(DATASET, EMB_TYPE), allow_pickle=True)
    print(
        "trn_point_features.shape, tst_point_features.shape, label_features.shape",
        trn_point_features.shape,
        tst_point_features.shape,
        label_features.shape)

    trn_X_Y = data_utils.read_sparse_file(
        "{}/trn_X_Y.txt".format(DATASET), force_header=True)

    tst_X_Y = data_utils.read_sparse_file(
        "{}/tst_X_Y.txt".format(DATASET), force_header=True)

    tst_valid_inds, trn_X_Y, tst_X_Y_trn, tst_X_Y_val, node_features, valid_tst_point_features, label_remapping, \
    adjecency_lists, NUM_TRN_POINTS = prepare_data(
        trn_X_Y, tst_X_Y, trn_point_features, tst_point_features, label_features,
        trn_point_titles, tst_point_titles, label_titles, args, logger)

    hard_negs = [[] for i in range(node_features.shape[0])]

    print("trn_X_Y.shape, tst_X_Y_trn.shape, tst_X_Y_val.shape",
          trn_X_Y.shape, tst_X_Y_trn.shape, tst_X_Y_val.shape)


    # ====================== DEFINING GRAPH ==========================

    graph = Graph(node_features, adjecency_lists, args["random_shuffle_nbrs"])

    params = create_params_dict(
        args,
        node_features,
        trn_X_Y,
        graph,
        NUM_TRN_POINTS)
    logger.info("params= %s", params)

    # ==============================   M1/Phase1 Training(with random negatives) ================

    head_net = JobXMLCBase(params["num_labels"], params["hidden_dims"], params["devices"],
                          params["feature_dim"], params["fanouts"], params["graph"], params["embed_dims"], args["encoder"])

    head_optimizer = torch.optim.Adam(
        [{'params': [head_net.classifier.classifiers[0].attention_weights], 'lr': params["attention_lr"]},
         {"params": [param for name, param in head_net.named_parameters() if
                     name != "classifier.classifiers.0.attention_weights"], "lr": params["lr"]}], lr=params["lr"])

    # getting error if we try to create a validation set - check why later on. For now not keeping a validation set!

    val_data = create_validation_data(valid_tst_point_features, label_features, tst_X_Y_val,
                                      args, params, TST_TAKE, head_net) # val_labels, val_loader dictionary

    # training loop
    warnings.simplefilter('ignore')

    head_criterion = torch.nn.BCEWithLogitsLoss(reduction=params["reduction"])
    logger.info("Model parameters: ", params)
    logger.info("Model configuration: ", str(head_net))

    head_train_dataset = DatasetGraph(trn_X_Y, hard_negs)  # Right now hard_negs are empty
    logger.info('Dataset Loaded')

    hc = GraphCollator(
        head_net,
        params["num_labels"],
        params["num_random_samples"],
        num_hard_neg=0)
    logger.info('Collator created')

    head_train_loader = torch.utils.data.DataLoader(
        head_train_dataset,
        batch_size=params["batch_size"],
        num_workers=10,
        collate_fn=hc,
        shuffle=True,
        pin_memory=False
    )

    inv_prop = xc_metrics.compute_inv_propesity(trn_X_Y, args["A"], args["B"])

    head_net.move_to_devices()

    if (args["mpt"] == 1):
        scaler = torch.cuda.amp.GradScaler()

    train(args, params, head_net, head_train_loader, val_data, head_criterion, head_optimizer,trial, tst_X_Y_trn, inv_prop)

    # should be kept as how many we want to test on
    params["num_tst"] = tst_X_Y_val.shape[0]

    # ============================================== SAVING MODEL ========================

    if (args["save_model"] == 1):
        model_dir = "{}/models/GraphXMLModel{}_{}".format(DATASET, RUN_TYPE, args["name"])

        if not os.path.exists(model_dir):
            print("Making model dir...")
            os.makedirs(model_dir)

        torch.save(
            head_net.state_dict(),
            os.path.join(
                model_dir,
                "model_state_dict.pt"))
        with open(os.path.join(model_dir, "model_params.pkl"), "wb") as fout:
            pickle.dump(params, fout, protocol=4)
        with open(os.path.join(model_dir, "predictions_20"), 'w') as f:
            pass

        with open(os.path.join(model_dir, 'commandline_args.txt'), 'w') as f:
            json.dump(args["__dict__"], f, indent=2)

    if (params["num_HN_epochs"] <= 0):
        print("Accuracies with graph embeddings to shortlist:")
        test(model_dir, args, params, head_net, RUN_TYPE, node_features, valid_tst_point_features, adjecency_lists,
             NUM_TRN_POINTS, label_remapping,
             label_features, tst_X_Y_val, tst_X_Y_trn)
        sys.exit(
            "You have chosen not to fine tune classifiers using hard negatives by providing num_HN_epochs <= 0")

    logger.info("==================================================================")

    # ================================   M4/Phase2 Training(with hard negatives) ===========================

    logger.info("params= %s", params)

    logger.info("******  Starting HN fine tuning of calssifiers  ******")

    prediction_shortlist_trn = sample_hard_negatives(
        head_net, label_remapping, trn_X_Y.shape[0], params)

    head_criterion = torch.nn.BCEWithLogitsLoss(reduction=params["reduction"])
    logger.info("Model parameters: ", params)

    head_train_dataset = DatasetGraph(trn_X_Y, prediction_shortlist_trn)
    logger.info('Dataset Loaded')


    head_optimizer = torch.optim.Adam(
        [{'params': [head_net.classifier.classifiers[0].attention_weights], 'lr': params["attention_lr"]},
         {"params": [param for name, param in head_net.classifier.named_parameters() if
                     name != "classifiers.0.attention_weights"], "lr": params["lr"]}], lr=params["lr"])

    hc = GraphCollator(
        head_net,
        params["num_labels"],
        0,
        num_hard_neg=params["num_HN_shortlist"])
    logger.info('Collator created')

    head_train_loader = torch.utils.data.DataLoader(
        head_train_dataset,
        batch_size=params["batch_size"],
        num_workers=6,
        collate_fn=hc,
        shuffle=True,
        pin_memory=True
    )

    inv_prop = xc_metrics.compute_inv_propesity(trn_X_Y, args["A"], args["B"])

    head_net.move_to_devices()

    if (args["mpt"] == 1):
        scaler = torch.cuda.amp.GradScaler()

    params["adjust_lr_epochs"] = np.arange(0, params["num_HN_epochs"], 4)
    params["num_epochs"] = params["num_HN_epochs"]

    train(args, params, head_net, head_train_loader, val_data, head_criterion, head_optimizer,logger,trial, tst_X_Y_trn, inv_prop)



    print("==================================================================")
    print("Accuracies with graph embeddings to shortlist:")
    params["num_tst"] = tst_X_Y_val.shape[0]
    recall_5 = test(model_dir, args, params, head_net, RUN_TYPE, node_features, valid_tst_point_features,
                    adjecency_lists, NUM_TRN_POINTS, label_remapping,
                    label_features, tst_X_Y_val, tst_X_Y_trn)

    wandb.run.summary["final recall@5"] = recall_5
    wandb.run.summary["state"] = "completed"
    wandb.finish(quiet=True)

    return recall_5
