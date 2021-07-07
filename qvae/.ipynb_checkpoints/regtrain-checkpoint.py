import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import pandas as pd
import math, random, sys
from optparse import OptionParser
from collections import deque

from jtnn import *
import rdkit
    
lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path")
parser.add_option("-u", "--train_prop", dest="train_prop_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-s", "--save_dir", dest="save_path")
parser.add_option("-m", "--modelvae", dest="modelvae_path", default=None)
parser.add_option("-r", "--modelreg", dest="modelreg_path", default=None)
parser.add_option("-b", "--batch", dest="batch_size", default=40)
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-q", "--lr", dest="lr", default=1e-3)
parser.add_option("-g", "--lr_reg", dest="lr_reg", default=1e-3)
parser.add_option("-e", "--stereo", dest="stereo", default=1)
opts,args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
lr = float(opts.lr)
lr_reg = float(opts.lr_reg)
stereo = True if int(opts.stereo) == 1 else False

model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)

regressor = nn.Sequential(
    nn.Linear(latent_size, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Linear(1024, 1),
)

if opts.modelvae_path is not None:
    model.load_state_dict(torch.load(opts.modelvae_path))
else:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

model = model.cuda()
regressor = regressor.cuda()

if opts.modelreg_path is not None:
    regressor.load_state_dict(torch.load(opts.modelreg_path))
else:
    for param in regressor.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
print("Model #Params: %dK" % (sum([x.nelement() for x in [*model.parameters(), *regressor.parameters()]]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.3)

optimizer_reg = optim.Adam(regressor.parameters(), lr=lr_reg)
scheduler_reg = lr_scheduler.ExponentialLR(optimizer_reg, 0.4)

fLoss = nn.MSELoss()
dataset = PropDataset(opts.train_path, opts.train_prop_path)
MAX_EPOCH = 10
PRINT_ITER = 20

dataset_size = len(dataset)
size = len(dataset)*MAX_EPOCH
pbar = tqdm(range(MAX_EPOCH))


verbose_df = pd.DataFrame(columns=["Step", "MSE Loss", "LR_enc", "LR_reg"])

for epoch in pbar:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3, collate_fn=lambda x:x, drop_last=True)
    loss  = 0
    
    for it, batch in enumerate(dataloader):
        tree, prop = zip(*batch)

        try:
            model.zero_grad()
            regressor.zero_grad()
            
            features = model.get_embeddings(list(tree))
            prediction = regressor(features)

            wloss = fLoss(prediction, torch.cuda.FloatTensor(prop)[:, None])
            wloss.backward()
            
            optimizer.step()
            optimizer_reg.step()

        except Exception as e:
            continue

        loss += float(wloss)
        
        if (it + 1) % PRINT_ITER == 0:
            loss = loss / PRINT_ITER


            verbose_df = verbose_df.append(pd.Series([it,
                                                      loss,
                                                      scheduler.get_last_lr()[0],
                                                      scheduler_reg.get_last_lr()[0]], index=verbose_df.columns), ignore_index=True)

            pbar.set_description("Epoch progress: %.2f General progress: %.2f MSE Loss: %f Learning rates ( encoder: %.6f, regressor: %.6f )" % \
                                 ( (it+1)*len(batch)*100/dataset_size, ((it+1)*len(batch) + dataset_size*epoch)*100/size, loss, scheduler.get_last_lr()[0], scheduler_reg.get_last_lr()[0]))

            loss = 0

            verbose_df.to_csv(opts.save_path +"/verbose.csv")

            torch.save(model.state_dict(), opts.save_path + "/model")
            torch.save(regressor.state_dict(), opts.save_path + "/model_regression")
            
    scheduler.step()
    scheduler_reg.step()
    torch.save(model.state_dict(), opts.save_path + "/model.epoch-" + str(epoch))
    torch.save(regressor.state_dict(), opts.save_path + "/model_regression.epoch-" + str(epoch))