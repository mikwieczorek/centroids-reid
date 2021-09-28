# encoding: utf-8
"""
Partially based on work by:
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

Adapted and extended by:
@author: mikwieczorek
"""

from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .df1 import DF1
from .street2shop import Street2Shop

__factory = {
    "market1501": Market1501,
    "dukemtmcreid": DukeMTMCreID,
    "df1": DF1,
    "street2shop": Street2Shop,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
