# ==============================================================================
# sinc(i) - http://sinc.unl.edu.ar/
# L. Bugnon, C. Yones, J. Raad, M. Gerard, M. Rubiolo, G. Merino, M. Pividori,
# L. Di Persia, D.H. Milone, and G. Stegmayer.
# lbugnon@sinc.unl.edu.ar
# ==============================================================================
import numpy as np
import torch
import pickle
from sklearn.utils import shuffle


class Dataloader():
    """
    Class to manage dataset and its partitions.
    """
    def __init__(self, data_dir, search_pair, device):

        self.search_pair = search_pair

        print("Loading %s-%s data..." % (search_pair[0], search_pair[1]))

        self.device = device
        
        # Embedded articles 
        self.data = pickle.load(open("%sembedding.pk" % (data_dir), "rb"))

        self.articles = pickle.load(open("%sarticles.pk" % data_dir, "rb"))

        # All possible combinations of search_pair words in the dataset that have at least one occurence.
        self.cases = pickle.load(open("%scases_%s_%s.pk" % (data_dir, search_pair[0], search_pair[1]), "rb"))
        
        print("Dataset ready with %d cases (%d positives)." % (len(self.cases), sum(self.cases["label"])))

        self.entities = ["gene", "drug", "text", "disease", "cell line"]

        self.cases = shuffle(self.cases)

    def get_labels(self, ind=None):
        """
        Get labels contained in ind. If ind=None, get all labels of the partition
        """
        if ind is not None:
            return self.cases["label"].values[ind]
        return self.cases["label"].values

    def get_articles(self, ind=np.array([])):
        """
        Get set of articles included in ind.
        """
        if np.size(ind) > 0:
            return self.cases["article"].values[ind]
        return self.cases["article"].values

    def get_batch(self, ind):
        """
        Returns data tensor and labels ready for training using the cases contained in ind.
        """
        data = torch.zeros((len(ind), self.data.shape[1], self.data.shape[2]))
        labels = torch.zeros((len(ind)))
        cases = self.cases.iloc[ind]
        
        for k, i in enumerate(ind):

            art=self.get_articles(i)
            data[k,:,:]=self.data[self.articles.index(art),:,:]

            case=self.cases.iloc[i]
            labels[k]=case["label"]
            
            terms=[l+1 for l in range(len(self.search_pair))] 

            for obj in terms:  # term1,term2
                for pos in case["pos%d" % obj]:  # positions in the article
                    if pos < self.data.shape[2]:
                        row = self.entities.index(case["target%d" % obj])
                        data[k, -1, pos] = 1
            
        data = data.float().to(self.device)
        labels = labels.float().to(self.device)

        return data, labels, cases

