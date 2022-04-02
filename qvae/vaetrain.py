import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque
import pandas as pd
import numpy as np

from jtnn import *
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-s", "--save_dir", dest="save_path")
parser.add_option("-m", "--model", dest="model_path", default=None)
parser.add_option("-b", "--batch", dest="batch_size", default=40)
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-z", "--beta", dest="beta", default=1.0)
parser.add_option("-q", "--lr", dest="lr", default=1e-3)
parser.add_option("-e", "--stereo", dest="stereo", default=1)
opts,args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
beta = float(opts.beta)
lr = float(opts.lr)
stereo = True if int(opts.stereo) == 1 else False

model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)

if opts.model_path is not None:
    model.load_state_dict(torch.load(opts.model_path))
else:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant(param, 0)
        else:
            nn.init.xavier_normal(param)

model = model.cuda()
print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))


optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)

dataset = MoleculeDataset(opts.train_path)

MAX_EPOCH = 7
PRINT_ITER = 100
    
    

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))


verbose_df = pd.DataFrame(columns=["Step", "Loss", "Beta", "KL", "Word", "Topo", "Assm", "Steo", "PNorm", "GNorm", "LR"])

for epoch in range(MAX_EPOCH):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x:x, drop_last=True)

    loss, word_acc,topo_acc,assm_acc,steo_acc = 0, 0,0,0,0

    for it, batch in enumerate(dataloader):
        for mol_tree in batch:
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)
                    node.cand_mols.append(node.label_mol)

        try:
            model.zero_grad()
            wloss, kl_div, wacc, tacc, sacc, dacc = model(batch, beta)
            wloss.backward()
            optimizer.step()
        except Exception as e:
            print(e)
            continue
        loss += float(wloss.item())
        word_acc += wacc
        topo_acc += tacc
        assm_acc += sacc
        steo_acc += dacc

        if (it + 1) % PRINT_ITER == 0:
            loss = loss / PRINT_ITER * 100
            word_acc = word_acc / PRINT_ITER * 100
            topo_acc = topo_acc / PRINT_ITER * 100
            assm_acc = assm_acc / PRINT_ITER * 100
            steo_acc = steo_acc / PRINT_ITER * 100
            
            verbose_df = verbose_df.append(pd.Series([it,
                                                      loss,
                                                      beta, 
                                                      kl_div,
                                                      word_acc,
                                                      topo_acc,
                                                      assm_acc,
                                                      steo_acc,
                                                      param_norm(model),
                                                      grad_norm(model),
                                                      scheduler.get_last_lr()[0]], index=verbose_df.columns), ignore_index=True)
            sys.stdout.flush()
            verbose_df.to_csv(opts.save_path +"verbose.csv")

            print("Loss: %.3f, KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f" % (loss, kl_div, word_acc, topo_acc, assm_acc, steo_acc))
            torch.save(model.state_dict(), opts.save_path + "/model")
            loss, word_acc,topo_acc,assm_acc,steo_acc = 0,0,0,0,0
            sys.stdout.flush()
            
        if (it + 1) % 15000 == 0: #Fast annealing
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_last_lr()[0])

    scheduler.step()
    print("learning rate: %.6f" % scheduler.get_last_lr()[0])
    torch.save(model.state_dict(), opts.save_path + "/model.epoch-" + str(epoch))

