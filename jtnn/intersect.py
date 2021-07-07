import rdkit
import rdkit.Chem as Chem
import copy
from tqdm import tqdm
from chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, enum_assemble, decode_stereo
from mol_tree import *

if __name__ == "__main__":
    import sys
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    qdb9_train = '../qvae/data/qdb9/train.txt'
    qdb9_test = '../qvae/data/qdb9/test.txt'
    cset = set()
    for i, line in tqdm(enumerate(sys.stdin)):
        try:
            smiles = line.split()[0]
            mol = MolTree(smiles)
            for c in mol.nodes:
                cset.add(c.smiles)
        except IndexError as e:
            print("index {} {}, Error: {}".format(i, line, e))
            pass
        
    qdb9_train_zinc = '../qvae/data/qdb9/train_zinc_only.txt'
    
    qdb9_test_zinc = '../qvae/data/qdb9/test_zinc_only.txt'
    
    qdb9_full_zinc = '../qvae/data/qdb9/all_zinc_only.txt'
    
    with open(qdb9_train_zinc, 'w') as train:
        with open(qdb9_test_zinc, 'w') as test:
            with open(qdb9_full_zinc, 'w') as full:
                with open(qdb9_train, "r") as qdb9_train:
                    
                    for i,line in tqdm(enumerate(qdb9_train.readlines())):
                        try:
                            is_valid = True
                            smiles = line.split()[0]
                            mol = MolTree(smiles)
                            for c in mol.nodes:
                                if not c.smiles in  cset:
                                    is_valid = False
                                    break
                                else:
                                    pass
                        except IndexError as e:
                            print("index {}, Error: {}".format(i, e))
                            pass

                        if is_valid:
                            train.write(line)
                            full.write(line)
                
                with open(qdb9_test) as qdb9_test:
                    for i,line in tqdm(enumerate(qdb9_test.readlines())):
                        try:
                            is_valid = True
                            smiles = line.split()[0]
                            mol = MolTree(smiles)
                            for c in mol.nodes:
                                if not c.smiles in  cset:
                                    is_valid = False
                                    break
                                else:
                                    pass

                            if is_valid:
                                test.write(line)
                                full.write(line)
                        except IndexError as e:
                            print("index {}, Error: {}".format(i, e))
                            pass
