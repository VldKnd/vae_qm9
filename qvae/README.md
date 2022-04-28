# Molecule Generation
Suppose the repository is downloaded at `$PREFIX/icml18-jtnn` directory. First set up environment variables:
```
export PYTHONPATH=/workspace/icml18-jtnn_original
```
Our ZINC dataset is in `icml18-jtnn/data/zinc` (copied from https://github.com/mkusner/grammarVAE). 
We follow the same train/dev/test split as previous work. 

## Deriving Vocabulary 
Vocabulary is stored in data/merged/vocab.txt

## Training
We trained VAE model in two phases:
1. Training without KL regularization term.
Pretrain our model as follows (with hidden state dimension=450, latent code dimension=56, graph message passing depth=3):
```
python3 pretrain.py --train ./data/merged/train.txt --vocab ./data/merged/vocab.txt \
--hidden * --depth * --latent * --batch * \
--save_dir *
```
The final model is saved at */model.iter-2

2. Train out model with KL regularization, with constant regularization weight $beta$
```
python3 vaetrain.py --train ./data/merged/train.txt --vocab ./data/merged/vocab.txt \
--hidden * --depth * --latent * --batch * --lr * --beta * \
--model path_to_pretrained_model/model.iter-2 --save_dir path_to_save_model/
```

## Testing
Molecule reconstruction, run  
```
python3 reconstruct.py --test ./data/merged/test.txt --vocab ./data/merged/vocab.txt \
--hidden * --depth * --latent * \
--model path_to_model/model.iter-6
```
Parameters should match with model stored in "path_to_model" folder
Replace `test.txt` with `valid.txt` to test the validation accuracy (for hyperparameter tuning).
## Training Regression
```
 python3 regtrain.py --train ./data/merged/reg_data/train_mols.txt --train_prop ./data/merged/reg_data/train_homo.txt --vocab ./data/merged/vocab.txt \
 --hidden * --depth * --latent * --batch * --lr * \
 --modelvae path_to_model/model.final --save_dir path_to_save_dir/
```
## Testing Regression
```
 python3 prop_test.py --test ./data/merged/reg_data/train_mols.txt --test_prop ./data/merged/reg_data/train_lumo.txt --vocab ./data/merged/vocab.txt --hidden * --depth * --latent * --batch * --modelvae path_to_model/model.final --save_dir path_to_saved_regression_head/model_regression.iter-0
```
Replace `test.txt` with `valid.txt` to test the validation accuracy (for hyperparameter tuning).
