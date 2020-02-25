# ==============================================================================
# sinc(i) - http://sinc.unl.edu.ar/
# L. Bugnon, C. Yones, J. Raad, M. Gerard, M. Rubiolo, G. Merino, M. Pividori,
# L. Di Persia, D.H. Milone, and G. Stegmayer.
# lbugnon@sinc.unl.edu.ar
# ==============================================================================
import os
import random
import shutil
import torch
import numpy as np
from src.sampler import Sampler
from src.dataloader import Dataloader
from src.model import Model
from src.logger import Logger

# Global params =======================
device = torch.device("cuda")
n_folds = 10
n_batch = 64

res_dir = "results_colab/"
shutil.rmtree(res_dir, ignore_errors=True)
os.mkdir(res_dir)

logger = Logger(res_dir)

for entity in ["gene", "drug"]:

    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)

    logger.start("test_%s_mutation" % entity)
    logger.log("entity\tfold\ttest_AUC\ttest_F1\ttest_precision\ttest_sensitivity\n")
    search_pair = [entity, "mutation"]

    dloader = Dataloader(data_dir="data/", device=device, search_pair=search_pair)

    sampler_list=[]
    for fold in range(n_folds):
        sampler_list.append(Sampler(dloader.get_articles(), dloader.get_labels(),
                                    n_batch=n_batch, fold=fold, nfold=n_folds))

    fold = 0
    while fold < n_folds:
        
        model = Model(res_dir, n_batch=n_batch, device=device)

        msg = "iter\ttrain_loss\ttrain_AUC\ttrain_F1\n"
        logger.log(msg, "train_%s_%02d" % (entity, fold))

        for it in range(model.n_epochs):
            train_loss = 0
            data, labels, _ = dloader.get_batch(sampler_list[fold].batch_ind("train"))
            train_loss, train_auc, train_f1 = model.train(data, labels)

            print('{} \t {:.4f} \t {:.4f} \t {:.4f}'.format(it, np.round(train_loss, 4), np.round(train_auc, 4),
                                                            np.round(train_f1, 4)))
        if train_auc < 0.8:  # train failed
            continue
        _, test_auc, test_f1, test_pre, test_rec = model.test(sampler_list[fold].batch_ind("test"), dloader)
        
        msg = "%s\t%d\t%.3f\t%.3f\t%.3f\t%.3f\n" % (entity, fold, test_auc, test_f1, test_pre, test_rec)
        logger.log(msg, "test_%s_mutation" % entity)
        fold += 1
