# encoding: utf-8
"""
@author: mikwieczorek
"""

import os

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only


class ModelCheckpointPeriodic(Callback):
    """
    ModelCheckpoint handler can be used to periodically save objects to disk.
    
    Args:
        dirname (str):
            Directory path where objects will be saved.
        filename_prefix (str):
            Prefix for the filenames to which objects will be saved. See Notes
            for more details.
        save_interval (int, optional):
            if not None, objects will be saved to disk every `save_interval` calls to the handler.
            Exactly one of (`save_interval`, `score_function`) arguments must be provided.
        n_saved (int, optional):
            Number of objects that should be kept on disk. Older files will be removed.
        atomic (bool, optional):
            If True, objects are serialized to a temporary file,
            and then moved to final destination, so that files are
            guaranteed to not be damaged (for example if exception occures during saving).
        create_dir (bool, optional):
            If True, will create directory 'dirname' if it doesnt exist.
        save_as_state_dict (bool, optional):
            If True, will save only the `state_dict` of the objects specified, otherwise the whole object will be saved.
    """

    def __init__(self, dirname, filename_prefix,
                 save_interval=None,
                 n_saved=1,
                 create_dir=True):

        super().__init__()
        self._dirname = os.path.expanduser(dirname)
        self._fname_prefix = filename_prefix
        self._n_saved = n_saved
        self._save_interval = save_interval
        self._saved = []  # list of saved filenames

        if create_dir:
            os.makedirs(dirname, exist_ok=True)

        # Ensure that dirname exists
        if not os.path.exists(dirname):
            raise ValueError("Directory path '{}' is not found.".format(dirname))

    @rank_zero_only
    def on_epoch_end(self, trainer, pl_module):
        assert trainer.is_global_zero, f'Periodic model checkpointer should only save from process number 0! Got: {trainer.global_rank}'

        current_epoch = trainer.current_epoch 

        if (current_epoch % self._save_interval) != 0:
            return

        if (len(self._saved) <= self._n_saved):
            fname = '{}_{}.pth'.format(self._fname_prefix, current_epoch)
            path = os.path.join(self._dirname, fname)
            trainer.save_checkpoint(filepath=path, weights_only=False)
            self._saved.append(path)

        if len(self._saved) > self._n_saved:
            path = self._saved.pop(0)
            if os.path.isfile(path):
                os.remove(path)
