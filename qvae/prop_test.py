import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

import math, random, sys
from optparse import OptionParser
from collections import deque

from jtnn import *
import rdkit
    
lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-u", "--test_prop", dest="test_prop_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--modelvae", dest="modelvae_path", default=None)
parser.add_option("-r", "--modelreg", dest="modelreg_path", default=None)
parser.add_option("-b", "--batch", dest="batch_size", default=40)
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-e", "--stereo", dest="stereo", default=1)
opts,args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
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

if opts.modelreg_path is not None:
    regressor.load_state_dict(torch.load(opts.modelreg_path))
else:
    for param in regressor.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    
model = model.cuda()
regressor = regressor.cuda()

model.eval()
regressor.eval()

print("Model #Params: %dK" % (sum([x.nelement() for x in [*model.parameters(), *regressor.parameters()]]) / 1000,))

fLoss = lambda x, y: torch.mean(torch.abs(y - x))
dataset = PropDataset(opts.test_path, opts.test_prop_path)
MAX_EPOCH = 2
PRINT_ITER = 10


dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x:x, drop_last=True)
loss = 0

size = len(dataset)
pbar = tqdm(dataloader)
for it, batch in enumerate(pbar):
    tree, prop = zip(*batch)

    try:
        features = model.get_embeddings(list(tree))
        prediction = regressor(features)

        wloss = fLoss(prediction, torch.cuda.FloatTensor(prop)[:, None])
        
    except Exception as e:
        print(e)

    loss += float(wloss)

    if (it + 1) % PRINT_ITER == 0:
        pbar.set_description("Progress: %.2f MAE Loss: %.6f" % ( (it+1)*len(batch)*100/size, loss*100/((it + 1)*len(batch)) ) )
