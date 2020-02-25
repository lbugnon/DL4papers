# ==============================================================================
# sinc(i) - http://sinc.unl.edu.ar/
# L. Bugnon, C. Yones, J. Raad, M. Gerard, M. Rubiolo, G. Merino, M. Pividori,
# L. Di Persia, D.H. Milone, and G. Stegmayer.
# lbugnon@sinc.unl.edu.ar
# ==============================================================================
import numpy as np
import random


class Sampler():
    """Partition sampler used for validation."""
    def __init__(self, articles, labels, n_batch, fold, nfold,
                 partition_size=[.9, 0, .1]):
        self.n_batch = n_batch
        partsize = partition_size

        L = len(labels)
        pos_ind = np.where(labels == 1)[0]
        neg_ind = np.where(labels == 0)[0]

        def rotate(l, n):
            return np.concatenate((l[-n:], l[:-n]))
        pos_ind = rotate(pos_ind, fold*len(pos_ind) // nfold)

        random.shuffle(neg_ind)
        n_pos = len(pos_ind)

        I = int(partsize[0]*n_pos)
        self.train_ind = np.concatenate((pos_ind[:I], neg_ind[:I]))

        II = I + int((partsize[1]) * n_pos)
        self.optim_ind = np.concatenate((pos_ind[I:II], neg_ind[I:II]))

        end = n_pos
        self.test_ind = np.concatenate((pos_ind[II:], neg_ind[II:end]))

        # add unused negative samples to train
        self.train_ind = np.concatenate((self.train_ind, neg_ind[end:]))
        self.labels = labels

    def batch_ind(self, part):
        ind = getattr(self, part+"_ind")
        if part != "train":
            return ind

        random.shuffle(ind)
        pSamples = ind[np.where(self.labels[ind] == 1)[0][:int(self.n_batch/2)]]
        nSamples = ind[np.where(self.labels[ind] == 0)[0][:int(self.n_batch/2)]]

        return np.concatenate((pSamples, nSamples))
