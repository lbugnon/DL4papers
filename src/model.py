# ==============================================================================
# sinc(i) - http://sinc.unl.edu.ar/
# L. Bugnon, C. Yones, J. Raad, M. Gerard, M. Rubiolo, G. Merino, M. Pividori,
# L. Di Persia, D.H. Milone, and G. Stegmayer.
# lbugnon@sinc.unl.edu.ar
# ==============================================================================
from torch.autograd import Variable
import torch
import pandas as pd
import sklearn.metrics as metrics
from src.network import network


class Model:
    """
    Class to define the deep learning model. It includes training and testing methods, parameter optimization and loss 
    reports.
    """
    n_epochs = 570
    
    def __init__(self, out_dir, n_batch, device, hyperparams=None):
        self.device = device
        self.n_batch = n_batch
        self.net = network(hyperparams=hyperparams)
        self.net = self.net.to(device)

        self.optimizer = torch.optim.Adadelta(self.net.parameters())
        self.criterion = torch.nn.BCELoss()

        self.out_dir = out_dir

    def train(self, data, labl):
        """
        Train model with data and labl, and return training metrics
        """
        self.net.train()
        self.optimizer.zero_grad()
        pred = self.net(Variable(data)).squeeze()
        loss = self.criterion(pred, Variable(labl))
        loss.backward()
        self.optimizer.step()
        auc, f1, pre, rec = self.get_error(labl.detach().cpu().numpy(),
                                      pred.detach().cpu().numpy())
        return loss.item(), auc, f1

    def test(self, ind, dloader, predict=False):
        """
        Evaluate the model with ind indexed data from dloader. Error metrics are returned.
        """
        self.net.eval()
        out = torch.tensor([])
        heatmap = torch.tensor([])
        cases = pd.DataFrame()

        for b in range(len(ind)//self.n_batch+1):
            if (b+1)*self.n_batch <len(ind):
                data, labels, bcases = dloader.get_batch(ind[b*self.n_batch:(b+1)*self.n_batch])
            else:
                data, labels, bcases = dloader.get_batch(ind[b*self.n_batch:])
            if len(data) == 0:
                break

            if len(out) == 0:
                out = self.net(Variable(data)).detach().cpu()
                #heatmap = self.net.heatmap(Variable(data)).detach().cpu()
                cases = bcases.copy()
                if len(out.shape) > 1:
                    out = out.squeeze()
                ref = labels.detach().cpu()
            else:
                out2 = self.net(Variable(data)).detach().cpu()
                #heatmap2 = self.net.heatmap(Variable(data)).detach().cpu()
                if len(out2.shape) > 1:
                    out2 = out2.squeeze()
                out = torch.cat((out, out2))
                #heatmap = torch.cat((heatmap,heatmap2))
                ref = torch.cat((ref, labels.detach().cpu()))
                cases = cases.append(bcases)
        loss = self.criterion(out, ref)

        if not predict:
            auc, f1, pre, rec = self.get_error(ref.numpy(), out.numpy())
            return loss.item(), auc, f1, pre, rec

        return cases, out, ref, heatmap

    def get_error(self, ref, out):
        """
        Compute error metrics
        """
        auc = metrics.roc_auc_score(ref, out)
        pre, rec, th = metrics.precision_recall_curve(ref, out)
        f1max = 0
        for p, r in zip(pre, rec):
            if p+r == 0:
                continue
            f1 = 2*p*r/(p+r)
            if f1 > f1max:
                f1max = f1
                premax = p
                recmax = r
        return auc, f1max, premax, recmax
