# encoding: utf-8
"""
@author: mikwieczorek
"""

from .bases import COCODatasetBase


class Street2Shop(COCODatasetBase):
    """
    Dataset class to load COCO-format data into re-id model
    This class inherites from the parent class only to create
    separate name for the dataset
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
