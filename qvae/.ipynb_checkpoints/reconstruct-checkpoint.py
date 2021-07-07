import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from CGRtools import smiles

import math, random, sys
from optparse import OptionParser
from collections import deque

import rdkit
import rdkit.Chem as Chem

from jtnn import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-e", "--stereo", dest="stereo", default=1)
opts,args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
stereo = True if int(opts.stereo) == 1 else False

model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()
model.eval()

data = []
with open(opts.test_path) as f:
    for line in f:
        s = line.strip("\r\n ").split()[0]
        data.append(s)

acc = 1
tot = 1

size = len(data)
pbar = tqdm(data)

for i, _smiles in enumerate(pbar):
    try:
        mol = Chem.MolFromSmiles(_smiles)
        smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)

        dec_smiles = model.reconstruct(smiles3D)


        mol1 = smiles(dec_smiles)
        mol1.clean_stereo()
        mol1.canonicalize()

        mol2 = smiles(smiles3D)
        mol2.clean_stereo()
        mol2.canonicalize()

        if mol1 == mol2:
            acc += 1
        tot += 1

        pbar.set_description("Current element %i/%i score: %f" % (i, size, acc / tot))
        """
        dec_smiles = model.recon_eval(smiles3D)
        tot += len(dec_smiles)
        for s in dec_smiles:
            if s == smiles3D:
                acc += 1
        print acc / tot
        """
    except:

        pbar.set_description("Current element %i/%i score: %f" % (i, size, acc / tot))
