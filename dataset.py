import random
from scipy.misc import imread
import numpy as np
import os
from glob import glob

CHANNELS_TO_MODE = {
        0: None,
        1: 'L',
        3: 'RGB',
        4: 'RGBA',
}

def load_dataset(dir, ext='*', channels=0):
    paths = glob(os.path.join(dir, "*."+ext))
    return DataSet(paths, CHANNELS_TO_MOE[channels])

# There are some TF helpers we could use here under the hood to be more efficient:
# https://www.tensorflow.org/how_tos/reading_data/
class DataSet(object):

    def __init__(self, paths, mode):
        self.paths = paths
        self.mode = mode
        self.index_in_epoch = 0
        self.num_examples = len(paths)
        self.epochs_completed = 0

    def sample_img(self):
        return imread(self.paths[0], mode=self.mode)

    def next_batch(batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            self.epochs_completed += 1
            random.shuffle(self.paths)
            start = 0
            self.index_in_epoch = batch_size
        end = self.index_in_epoch
        imgs = [imread(path, mode=self.mode) for path in self.paths[start:end]]
        return np.array(imgs).astype(np.float32)
