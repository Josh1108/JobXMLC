import torch
from typing import Dict


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