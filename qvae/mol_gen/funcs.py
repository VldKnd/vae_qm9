import matplotlib.pyplot as plt
from CGRtools import smiles
from rdkit import Chem
from tqdm import tqdm
import torch
import rdkit

def latent_space_descent(variables, target, regressor, loss, opt, n_steps = 1):
    regressor.train()
        
    for i in range(n_steps):
        regressor.zero_grad()
        opt.zero_grad()
        
        prediction = regressor(variables)
        wloss = loss(prediction, target)
        wloss.backward()
        opt.step()
        
    return prediction.cpu().detach().numpy()

def latent_space_descent_two_proprieties(variables, target, regressor, loss, opt, n_steps = 1):
    regressor.eval()
    props = (torch.ones(variables.size()[0], 1)*target).cuda()
        
    for i in range(n_steps):
        regressor.zero_grad()
        opt.zero_grad()
        
        prediction = regressor(variables)
        wloss = loss(prediction, props)
        wloss.backward()
        opt.step()
        
    return prediction.cpu().detach().numpy()

def latent_space_descent_m(variables, target, regressor, loss, opt, n_steps = 1):
    regressor.train()
    props = (torch.ones(variables.size()[0], target.shape[0])*target).cuda()
        
    for i in range(n_steps):
        regressor.zero_grad()
        opt.zero_grad()
        
        prediction = regressor(variables)
        wloss = loss(prediction, props)
        wloss = wloss.sum(dim=1).mean(dim=0)
        wloss.backward()
        opt.step()
        
    return prediction.cpu().detach().numpy()
    
def init_latent_vector(size, example=None):
    if example is not None:
        return torch.tensor(example + torch.rand(size, device="cuda"), requires_grad=True)

    else:
        return torch.rand(size, device="cuda", requires_grad=True)

def show_molecule(mol):
    plt.figure(figsize=(18, 6))
    img = Chem.Draw.MolToImage(Chem.MolFromSmiles(mol))
    plt.axis("off")
    plt.imshow(img)
    plt.show()

def clean(smls):
    try:
        mol = smiles(smls)
        mol.clean_stereo()
        mol.canonicalize()
        return mol.__str__()
    except:
        return None

# def double_descent(variables,
#                    target,
#                    search_model,
#                    search_regression,
#                    regression_model,
#                    regression_regressor,
#                    loss,
#                    lr,
#                    n_steps = 1,
#                    n_iter = 1,
#                    verbose=True):
    
#     search_model.eval()
#     search_regression.eval()
#     regression_model.eval()
#     regression_regressor.eval()
#     try:
        
#         if verbose:
#             pbar = tqdm(range(n_iter))
            
#         else:
#             pbar = range(n_iter)
            
#         for j in pbar:
#             tree_vec, mol_vec = torch.split(variables, variables.size()[1]//2, 1)
#             s = []

#             for i, (tree, mol) in enumerate(zip(tree_vec, mol_vec)):
#                 s_string = search_model.decode(tree[None, :],mol[None, :], False)

#                 if s_string is not None:
#                     _s = smiles(search_model.decode(tree[None, :],mol[None, :], False))
#                     _s.clean_stereo()
#                     _s.canonicalize()
#                     s.append(_s.__str__())

#             s = list(filter(lambda x: Chem.MolFromSmiles(x) is not None, set(s)))
#             coeffs = regression_regressor(regression_model.encode_latent_mean(s))
#             idxes = torch.abs(coeffs - target).argsort(dim=0).flatten()
#             variables = variables[idxes[:10]].repeat((10, 1))
#             variables = (variables.detach() + torch.rand([variables.shape[0], 256], device="cuda"))
#             variables.requires_grad = True 
#             optimizer = optim.Adam([variables], lr=lr)

#             regressor.train()
#             props = (torch.ones(variables.size()[0], 1, device="cuda")*target).cuda()
#             for i in range(n_steps):
#                 search_regression.zero_grad()
#                 optimizer.zero_grad()

#                 prediction = search_regressor(variables)
#                 wloss = loss(prediction, props)
#                 wloss.backward()
#                 optimizer.step()
#         return variables
    
#     except:
#         return variables

# def get_metrics(model, molecules, ss, target_molecule, target_g):
#     """
#     Add best cosine similarity/ smallest L2 norm to hidden vectors as a metric 
#     Add distance between vocabularies 
#     Add cosine similarity of tree
#     Add cosine similarity of molecules
#     Add L2 distance for tree
#     Add L2 distance for molecules
#     Add Levenshtain distance
#     """
    
#     L2_best = ((molecules - target_molecule)**2).sum(dim=1).min().item()
#     Cosine_best = torch.nn.functional.cosine_similarity(target_molecule, molecules).max().item()
    
#     tree_vec, mol_vec = torch.split(molecules, molecules.size()[1]//2, 1)
#     tree_vec_target, mol_vec_target = torch.split(target_molecule, target_molecule.size()[1]//2, 1)
    
#     L2_best_tree = ((tree_vec - tree_vec_target)**2).sum(dim=1).min().item()
#     Cosine_best_tree = torch.nn.functional.cosine_similarity(tree_vec, tree_vec_target).max().item()
    
#     L2_best_mol = ((mol_vec - mol_vec_target)**2).sum(dim=1).min().item()
#     Cosine_best_mol = torch.nn.functional.cosine_similarity(mol_vec, mol_vec_target).max().item()
    
#     binary_smiles = []
#     for s in ss:
#         try:
#             mol = MolTree(s)
#             binary_h = [0]*vocab.size()
#             for c in mol.nodes:
#                 binary_h[vocab.get_index(c.smiles)] = 1

#             binary_smiles.append(binary_h)
#         except BaseException:
#             continue

#     binary_smiles = torch.tensor(binary_smiles, device = torch.device("cuda"))
    
#     target = [[0]*vocab.size()]
    
#     for n in target_g.nodes:
#         try:
#             target[0][vocab.get_index(n.smiles)] = 1
#         except KeyError:
#             continue
        
#     target = torch.tensor(target, device = torch.device("cuda"))
#     dice = (2*(binary_smiles*target).sum(1)/(binary_smiles.sum(1) + target.sum(1))).max().item()
#     tanimoto = ((binary_smiles*target).sum(1)/(((binary_smiles + target) > 0).sum(1))).max().item()
#     lev = np.min([Lev(target_g.smiles, s) for s in ss])
    
#     _, tree_vec_target, mol_vec_target = model.encode([tree_t])
#     tree_mean = model.T_mean(tree_vec_target)
#     tree_var = torch.exp(-torch.abs(model.T_var(tree_vec_target)))
#     mol_mean = model.G_mean(mol_vec_target)
#     mol_var = torch.exp(-torch.abs(model.G_var(mol_vec_target)))

#     mu, sigma = torch.cat([tree_mean,mol_mean], dim=1), torch.cat([tree_var,mol_var], dim=1)
    
#     H = 1/torch.sqrt(2*pi*sigma)
#     h = torch.exp(-(mu - molecules)**2/(2*sigma))/torch.sqrt(2*pi*sigma)
#     slutskiy = (h/H).mean(dim=1).max().item()
    
#     return [L2_best, Cosine_best, L2_best_tree, Cosine_best_tree, L2_best_mol, Cosine_best_mol, dice, tanimoto, lev, slutskiy]