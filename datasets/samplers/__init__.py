# encoding: utf-8
"""
@author: mikwieczorek
"""

from .distributed_pids_sampler import *

__factory = {
    'random_identity': RandomIdentitySampler
}

def get_names():
    return __factory.keys()

def get_sampler(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown sampler: {}".format(name))
    return __factory[name](**kwargs)
