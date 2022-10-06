import rdkit
import torch
from funcs import *
import pandas as pd
from torch import nn
import math, random, sys
PATH_TO_PROJECT = ""
sys.path.append(PATH_TO_PROJECT) ## To sepcify

from jtnn import *

from warnings import filterwarnings
filterwarnings("ignore")

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

## Model Loading
vocab_path = "data/merged/vocab.txt"
vocab = [x.strip("\r\n ") for x in open(vocab_path)]
cset = set(vocab)
vocab = Vocab(vocab)

hidden_size = 612
latent_size = 256
depth = 3
stereo = True
batch_size = 8192
random_state = 42

modelvae_path = 'reg_model/merged_qdb9_256_612_homo/decoder_bigger_batch/model'
regressorvae_path = 'reg_model/merged_qdb9_256_612_homo/model_regression'

model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
regressor = nn.Sequential(
    nn.Linear(latent_size, 4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, 1),
)

vocab_path = "data/merged/vocab.txt"
vocab = [x.strip("\r\n ") for x in open(vocab_path)]
cset = set(vocab)
vocab = Vocab(vocab)
scale = 27.21

columns = ["tree", "homo", "lumo", "gap"]

data = pd.read_csv("data/merged/all.txt")
df_val = pd.read_csv("data/merged/csv/val_reg.csv")
df_val.homo = df_val.homo*scale
df_val.lumo = df_val.lumo*scale
df_val.gap = df_val.gap*scale

model.load_state_dict(torch.load(modelvae_path))
regressor.load_state_dict(torch.load(regressorvae_path))

model = model.cuda()
regressor = regressor.cuda()
df = df_val

lines = []
size, cols = df.shape
pbar = tqdm(range(size))


for i in pbar:
    line = df.loc[i]
    mol_tree = MolTree(line.smiles)
    mol_tree.recover()
    mol_tree.assemble()
    lines.append([mol_tree, line.homo, line.lumo, line.gap])
    
tree_val = pd.DataFrame(lines, columns = columns)

X_val = pd.DataFrame(columns=[str(x) for x in range(latent_size)])
y_val = pd.DataFrame(columns=["homo", "lumo", "gap"])
df = tree_val
size, cols = df.shape

with torch.no_grad():
    batch_count = math.ceil(size/batch_size)
    pbar = tqdm(range(batch_count))
    
    for i in pbar:        
        _, tree_vec, mol_vec = model.encode(list(df[batch_size*i: batch_size*(i+1)]["tree"]))
        tree_mean = model.T_mean(tree_vec)
        mol_mean = model.G_mean(mol_vec)
        features = torch.cuda.FloatTensor(torch.cat([tree_mean,mol_mean], dim=1))
        X_val = X_val.append(pd.DataFrame(features.cpu().numpy(), columns=X_val.columns),ignore_index=True)
        y_val = y_val.append(df[batch_size*i: batch_size*(i+1)][["homo", "lumo", "gap"]],ignore_index=True)

### Homo value to be specified
homo = 0.1
lr_mol, n_steps = 0.01, 150
loss = nn.MSELoss(reduction="mean")

target = [homo]
size = [100, 256]

molecules = torch.rand(size, device=torch.device("cuda"))\
            *torch.tensor(X_val.std(), device=torch.device("cuda"))\
            +torch.tensor(X_val.mean(), device=torch.device("cuda"))

molecules.requires_grad = True

target = torch.FloatTensor(target)
target = target.to(torch.device("cuda"))
optimizer = torch.optim.Adam([molecules], lr=lr_mol)
props = latent_space_descent(molecules, target.repeat((100, 1)), regressor, nn.MSELoss(reduction="mean"), optimizer, 100)        
tree_vec, mol_vec = torch.split(molecules, molecules.size()[1]//2, 1)

s = set()

for i, (tree, mol) in enumerate(zip(tree_vec, mol_vec)):
    _s = model.decode(tree[None, :],mol[None, :], False)
    if _s is not None:
        _s = smiles(model.decode(tree[None, :],mol[None, :], False))
        _s.clean_stereo()
        _s.canonicalize()
        s.add(_s)

for _s in s:
    print(_s)
