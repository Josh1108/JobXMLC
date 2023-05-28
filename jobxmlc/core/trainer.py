from pyaml_env import parse_config
from jobxmlc import make_initial_embeddings
# flake8: noqa
from jobxmlc.registry import ENCODER_REGISTRY, DATA_FILTER_REGISTRY
from jobxmlc.core.utils import remove_key_from_dict, get_device,create_params_dict
import argparse
from typing import Dict
from jobxmlc.core.train_helper import data_loader,prepare_data,training_first_phase
from jobxlmc.core.data import Graph

def make_embeddings(encoder_parameters: Dict) -> None:
    encoder = ENCODER_REGISTRY[encoder_parameters["name"]](**remove_key_from_dict(encoder_parameters))
    encoder.embeddings_runner()
    return None


# def create_reporters(reporting: List[Dict], config_file: str):
#     reporting_hooks, reporting = merge_reporting_with_settings(reporting, {})
#     return create_reporting(reporting_hooks, reporting, {"config_file": config_file, "task": "eptest", "base_dir": "."})


def main():
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--config_file", default="config/spr2.yml")
    args = parser.parse_args()
    exp_params = parse_config(args.config_file)
    # make_embeddings(exp_params['encoder'])
    data_dict = data_loader(exp_params['model']['dataset_path'],exp_params['model']['embedding_path'])
    tst_valid_inds, trn_X_Y, tst_X_Y_trn, tst_X_Y_val, node_features, valid_tst_point_features, label_remapping, adjecency_lists, NUM_TRN_POINTS = prepare_data(data_dict,args=exp_params['model'])
    hard_negs = [[] for i in range(node_features.shape[0])]
    TST_TAKE = exp_params['model']['num_validation']
    graph = Graph(node_features, adjecency_lists, exp_params['model']['random_shuffle_nbrs'])
    params = create_params_dict(
        exp_params['model'],
        node_features,
        trn_X_Y,
        graph,
        NUM_TRN_POINTS)
    training_first_phase(params,trn_X_Y)


if __name__ == "__main__":
    main()