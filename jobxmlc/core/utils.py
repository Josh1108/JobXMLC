import torch
from typing import Dict
from scipy.sparse import csr_matrix

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