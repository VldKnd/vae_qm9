# Molecule Generation
Suppose the repository is downloaded at `$PREFIX/icml18-jtnn` directory. First set up environment variables:
```
export PYTHONPATH=/workspace/icml18-jtnn_original
```
Our ZINC dataset is in `icml18-jtnn/data/zinc` (copied from https://github.com/mkusner/grammarVAE). 
We follow the same train/dev/test split as previous work. 

## Deriving Vocabulary 
Vocabulary is stored in data/vocab.txt

## Training
We trained VAE model in two phases:
1. Training without KL regularization term.
Pretrain our model as follows (with hidden state dimension=450, latent code dimension=56, graph message passing depth=3):
```
ZINC
python3 pretrain.py --train ./data/zinc/train.txt --vocab ./data/vocab.txt \
--hidden 450 --depth 3 --latent 256 --batch 40 \
--save_dir pre_model/zinc_256/

QDB9
python3 pretrain.py --train ./data/qdb9/train.txt --vocab ./data/vocab.txt \
--hidden 612 --depth 3 --latent 256 --batch 40 \
--save_dir pre_model/qdb9_256_612/
```
The final model is saved at pre_model/zinc/model.iter-2

2. Train out model with KL regularization, with constant regularization weight $beta$
```
ZINC
python3 vaetrain.py --train ./data/zinc/train.txt --vocab ./data/vocab.txt \
--hidden 450 --depth 3 --latent 256 --batch 40 --lr 0.0007 --beta 0.005 \
--model pre_model/zinc_256/model.iter-2 --save_dir vae_model/zinc_256/

QDB9
python3 vaetrain.py --train ./data/qdb9/train.txt --vocab ./data/vocab.txt \
--hidden 612 --depth 3 --latent 256 --batch 40 --lr 0.0007 --beta 0.005 \
--model pre_model/qdb9_256_612/model.iter-2 --save_dir vae_model/qdb9_256_612/
```

## Testing
Molecule reconstruction, run  
```
python3 reconstruct.py --test ./data/zinc/test.txt --vocab ./data/vocab.txt \
--hidden 450 --depth 3 --latent 56 \
--model vae_model/zinc/model.iter-6
```

```
python3 reconstruct.py --test ./data/qdb9/test.txt --vocab ./data/vocab.txt \
--hidden 612 --depth 3 --latent 256 \
--model vae_model/qdb9_256_612_0.00007/model.iter-6
```

```
python reconstruct.py --test ./data/qdb9/new_qdb9/test_intersect_zinc.txt --vocab ./data/zinc/vocab.txt \
--hidden 450 --depth 3 --latent 256 \
--model vae_model/qdb9_zinc_256/model.iter-6-2000
```

Replace `test.txt` with `valid.txt` to test the validation accuracy (for hyperparameter tuning).
```
## Training Regression
Molecule reconstruction, run  

```
 python3 regtrain.py --train ./data/reg_data/train_mols.txt --train_prop ./data/reg_data/train_homo.txt --vocab ./data/zinc/vocab.txt \
 --hidden 450 --depth 3 --latent 56 --batch 40 --lr 0.0007 \
 --modelvae vae_model/paper_weights/model.final --save_dir reg_model/qdb9_zinc_paper_homo/
```

## Testing Regression
 python3 prop_test.py --test ./data/reg_data/train_mols.txt --test_prop ./data/reg_data/train_lumo.txt --vocab ./data/zinc/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 40 --modelvae vae_model/qdb9_zinc_256/model.iter-0 --save_dir reg_model/qdb9_zinc_256/model_regression.iter-0
 
Replace `test.txt` with `valid.txt` to test the validation accuracy (for hyperparameter tuning).
```

1. python3 regtrain.py --train ./data/reg_data/train_mols.txt --train_prop ./data/reg_data/train_lumo.txt --vocab ./data/zinc/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 40 --lr 0.0001 --lr_reg 0.0001 --modelvae vae_model/qdb9_zinc_256/model.iter-6 --save_dir reg_model/qdb9_zinc_256_lumo/

python3 regtrain.py --train ./data/reg_data/train_mols.txt --train_prop ./data/reg_data/train_lumo.txt --vocab ./data/zinc/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 40 --lr 0.00001 --lr_reg 0.00001 --modelvae reg_model/qdb9_zinc_256_lumo/model.final --modelreg reg_model/qdb9_zinc_256_lumo/model_regression.final --save_dir reg_model/qdb9_zinc_256_lumo/

python3 prop_test.py --test ./data/reg_data/train_mols.txt --test_prop ./data/reg_data/train_lumo.txt --vocab ./data/zinc/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 40 --modelvae reg_model/qdb9_zinc_256_lumo/model.final --modelreg reg_model/qdb9_zinc_256_lumo/model_regression.final

2. python3 regtrain.py --train ./data/reg_data/train_mols.txt --train_prop ./data/reg_data/train_homo.txt --vocab ./data/zinc/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 40 --lr 0.0001 --lr_reg 0.0001 --modelvae vae_model/qdb9_zinc_256/model.iter-6 --save_dir reg_model/qdb9_zinc_256_homo/

python3 regtrain.py --train ./data/reg_data/train_mols.txt --train_prop ./data/reg_data/train_homo.txt --vocab ./data/zinc/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 40 --lr 0.00001 --lr_reg 0.00001 --modelvae reg_model/qdb9_zinc_256_homo/model.final --modelreg reg_model/qdb9_zinc_256_homo/model_regression.final --save_dir reg_model/qdb9_zinc_256_homo/

python3 prop_test.py --test ./data/reg_data/train_mols.txt --test_prop ./data/reg_data/train_homo.txt --vocab ./data/zinc/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 40 --modelvae reg_model/qdb9_zinc_256_homo/model.final --modelreg reg_model/qdb9_zinc_256_homo/model_regression.final

3. python3 regtrain.py --train ./data/reg_data/train_mols.txt --train_prop ./data/reg_data/train_homo.txt --vocab ./data/zinc/vocab.txt --hidden 450 --depth 3 --latent 56 --batch 40 --lr 0.0001 --lr_reg 0.0001 --modelvae vae_model/paper_weights/model.final --save_dir reg_model/qdb9_zinc_paper_homo/

python3 regtrain.py --train ./data/reg_data/train_mols.txt --train_prop ./data/reg_data/train_homo.txt --vocab ./data/zinc/vocab.txt --hidden 450 --depth 3 --latent 56 --batch 40 --lr 0.00001 --lr_reg 0.00001 --modelvae reg_model/qdb9_zinc_paper_homo/model.final --modelreg reg_model/qdb9_zinc_paper_homo/model_regression.final --save_dir reg_model/qdb9_zinc_paper_homo/

python3 prop_test.py --test ./data/reg_data/train_mols.txt --test_prop ./data/reg_data/train_homo.txt --vocab ./data/zinc/vocab.txt --hidden 450 --depth 3 --latent 56 --batch 40 --modelvae reg_model/qdb9_zinc_paper_homo/model.final --modelreg reg_model/qdb9_zinc_paper_homo/model_regression.final

4. python3 regtrain.py --train ./data/reg_data/train_mols.txt --train_prop ./data/reg_data/train_homo.txt --vocab ./data/zinc/vocab.txt --hidden 450 --depth 3 --latent 56 --batch 40 --lr 0.0001 --lr_reg 0.0001 --modelvae vae_model/paper_weights/model.final --save_dir reg_model/qdb9_zinc_paper_lumo/

python3 regtrain.py --train ./data/reg_data/train_mols.txt --train_prop ./data/reg_data/train_lumo.txt --vocab ./data/zinc/vocab.txt --hidden 450 --depth 3 --latent 56 --batch 40 --lr 0.00001 --lr_reg 0.00001 --modelvae reg_model/qdb9_zinc_paper_lumo/model.final --modelreg reg_model/qdb9_zinc_paper_lumo/model_regression.final --save_dir reg_model/qdb9_zinc_paper_lumo/

python3 prop_test.py --test ./data/reg_data/train_mols.txt --test_prop ./data/reg_data/train_lumo.txt --vocab ./data/zinc/vocab.txt --hidden 450 --depth 3 --latent 56 --batch 40 --modelvae reg_model/qdb9_zinc_paper_lumo/model.final --modelreg reg_model/qdb9_zinc_paper_lumo/model_regression.final


# Training VAE on zinc+qdb9 prunned

python3 pretrain.py --train ./data/qdb9/prunned/qdb9+zinc/train_smiles.txt --val ./data/qdb9/prunned/qdb9+zinc/val_smiles.txt --vocab ./data/qdb9/prunned/vocab.txt \
--hidden 450 --depth 3 --latent 256 --batch 40 \
--save_dir pre_model/qdb9+zinc_prunned_256/


python3 pretrain.py --train ./data/qdb9/prunned/qdb9/train_smiles.txt --vocab ./data/qdb9/prunned/vocab.txt \
--hidden 450 --depth 3 --latent 256 --batch 40 \
--save_dir pre_model/qdb9_prunned_256/

python3 vaetrain.py --train ./data/qdb9/prunned/qdb9/train.txt --vocab ./data/qdb9/prunned/vocab.txt \
--hidden 450 --depth 3 --latent 256 --batch 40 --lr 0.0007 --beta 0.005 \
--model pre_model/qdb9_prunned_256/model.epoch-2 --save_dir vae_model/qdb9_prunned_256/


python3 vaetrain.py --train ./data/qdb9/prunned/qdb9+zinc/train.txt --vocab ./data/qdb9/prunned/vocab.txt \
--hidden 450 --depth 3 --latent 256 --batch 40 --lr 0.0007 --beta 0.005 \
--model pre_model/qdb9+zinc_prunned_256/model --save_dir vae_model/qdb9_zinc_prunned_256/



# Testing VAE on qdb9 prunned

python3 reconstruct.py --train ./data/qdb9/prunned/qdb9+zinc/test.txt --vocab ./data/qdb9/prunned/vocab.txt \
--hidden 450 --depth 3 --latent 256 \
--model vae_model/qdb9_prunned_256/model.iter-6

# Training regression on qdb9 prunned

python3 regtrain.py --train ./data/qdb9/prunned/qdb9/train_smiles.txt --train_prop ./data/qdb9/prunned/qdb9/train_lumo.txt --vocab ./data/qdb9/prunned/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 256 --lr 0.01 --lr_reg 0.1 --modelvae vae_model/qdb9_prunned_256/model.epoch-6 --save_dir reg_model/qdb9_prunned_256_lumo

python3 regtrain.py --train ./data/qdb9/prunned/qdb9/train_smiles.txt --train_prop ./data/qdb9/prunned/qdb9/train_homo.txt --vocab ./data/qdb9/prunned/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 256 --lr 0.01 --lr_reg 0.1 --modelvae vae_model/qdb9_prunned_256/model.epoch-6 --save_dir reg_model/qdb9_prunned_256_lumo

python3 regtrain.py --train ./data/qdb9/prunned/qdb9/train_smiles.txt --train_prop ./data/qdb9/prunned/qdb9/train_lumo.txt --vocab ./data/qdb9/prunned/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 256 --lr 0.01 --lr_reg 0.1 --modelvae vae_model/qdb9_zinc_prunned_256/model.epoch-6 --save_dir reg_model/qdb9_zinc_prunned_256_lumo

python3 regtrain.py --train ./data/qdb9/prunned/qdb9/train_smiles.txt --train_prop ./data/qdb9/prunned/qdb9/train_homo.txt --vocab ./data/qdb9/prunned/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 256 --lr 0.01 --lr_reg 0.1 --modelvae vae_model/qdb9_zinc_prunned_256/model.epoch-6 --save_dir reg_model/qdb9_zinc_prunned_256_homo

# Testing regression on qdb9 prunned

python3 prop_test.py --test ./data/qdb9/prunned/qdb9/test_smiles.txt --test_prop ./data/qdb9/prunned/qdb9/test_lumo.txt --vocab ./data/qdb9/prunned/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 40 --modelvae vae_model/qdb9_prunned_256/model.epoch-6 --modelreg reg_model/qdb9_prunned_256_lumo/model_regression

python3 prop_test.py --test ./data/qdb9/prunned/qdb9/test_smiles.txt --test_prop ./data/qdb9/prunned/qdb9/test_homo.txt --vocab ./data/qdb9/prunned/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 40 --modelvae vae_model/qdb9_prunned_256/model.epoch-6 --modelreg reg_model/qdb9_prunned_256_homo/model_regression

python3 prop_test.py --test ./data/qdb9/prunned/qdb9/test_smiles.txt --test_prop ./data/qdb9/prunned/qdb9/test_homo.txt --vocab ./data/qdb9/prunned/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 40 --modelvae vae_model/qdb9_zinc_prunned_256/model.epoch-6 --modelreg reg_model/qdb9_zinc_prunned_256_homo/model_regression

python3 prop_test.py --test ./data/qdb9/prunned/qdb9/test_smiles.txt --test_prop ./data/qdb9/prunned/qdb9/test_lumo.txt --vocab ./data/qdb9/prunned/vocab.txt --hidden 450 --depth 3 --latent 256 --batch 40 --modelvae vae_model/qdb9_zinc_prunned_256/model.epoch-6 --modelreg reg_model/qdb9_zinc_prunned_256_lumo/model_regression

# Testing reconstruction

python reconstruct.py --test ./data/qdb9/prunned/qdb9/test_smiles.txt --vocab ./data/qdb9/prunned/vocab.txt \
--hidden 450 --depth 3 --latent 256 \
--model vae_model/qdb9_zinc_prunned_256/model

python reconstruct.py --test ./data/qdb9/prunned/qdb9/test_smiles.txt --vocab ./data/qdb9/prunned/vocab.txt \
--hidden 450 --depth 3 --latent 256 \
--model vae_model/qdb9_prunned_256/model

python reconstruct.py --test ./data/qdb9/prunned/qdb9/test_smiles.txt --vocab ./data/qdb9/prunned/vocab.txt \
--hidden 450 --depth 3 --latent 256 \
--model pre_model/qdb9_prunned_256/model