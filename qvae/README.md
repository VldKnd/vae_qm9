In our work we have used a mix of [ZINC](http://zinc15.docking.org) and [QM9](http://quantum-machine.org/datasets/) datasets. Since the ZINC dataset have been used in previous works,
we use the same percentage of train/test/val split as in [Junction Tree Variational Autoencoder](https://arxiv.org/abs/1802.04364). Please note that **all the data used is provided in folder ./data**.
## Vocabulary 
Vocabulary used by our model is stored in data/merged/vocab.txt

## Training
We trained VAE model in two phases:
1. Training without KL regularization term.
```
pretrain.py [OPTIONS]
            --train [FILE_PATH] # Path to folder with smiles in txt format
            --vocab [FILE_PATH] # Path to vocabulary used for JT decomposition
            --hidden [INT] # Size of the hidden state size of the model
            --depth [INT] # Depth of message passing
            --latent [INT] # Size of latent representation of molecule
            --batch [INT] # Batch size
            --save_dir [FILE_PATH] # Path to store weights after trainning
```
We have trained the model on mix of QM9 and ZINC data, that can be found in `{$PROJECT_FOLDER}/qvae/data/merged/*`. 
To repeat the training with the parameters, mentioned in the paper, run hidden state dimension=450, latent code dimension=56, graph message passing depth=3):
```
python pretrain.py --train ./data/merged/train.txt \
                   --vocab ./data/merged/vocab.txt \
                   --hidden 612                    \
                   --depth 3                       \
                   --latent 256                    \
                   --batch 64                      \
                   --save-dir ./enc_model          \
```
The final model is saved at `./enc_model/model.iter-2`

2. Train out model with KL regularization, with constant regularization weight `beta`
```
vaetrain.py [OPTIONS]
            --train [FILE_PATH] # Path to folder with smiles in txt format
            --vocab [FILE_PATH] # Path to vocabulary used for JT decomposition
            --hidden [INT] # Size of the hidden state size of the model
            --depth [INT] # Depth of message passing
            --latent [INT] # Size of latent representation of molecule
            --batch [INT] # Batch size
            --beta [FLOAT] # Weight coefficient for KL Divergance
            --lr [FLOAT] # Learning rate (during the last stage is set to 1e-3
            --model [FILE_PATH] # Pre-trained encoder-decoder pair weights
            --save_dir [FILE_PATH] # Path to store weights after trainning
```
To get the provided weights, run
```
python vaetrain.py --train ./data/merged/train.txt  \
                   --vocab ./data/merged/vocab.txt  \
                   --hidden 612                     \
                   --depth 3                        \
                   --latent 256                     \
                   --batch 64                       \
                   --lr 0.007                       \
                   --beta 0.005                     \
                   --model ./enc_model/model.iter-2 \
                   --save_dir ./var_model           \
```
## Testing
To test molecule reconstruction, run
```
reconstruct.py [OPTIONS]
            --test [FILE_PATH] # Path to folder with smiles in txt format
            --vocab [FILE_PATH] # Path to vocabulary used for JT decomposition
            --hidden [INT] # Size of the hidden state size of the model
            --depth [INT] # Depth of message passing
            --latent [INT] # Size of latent representation of molecule
            --model [FILE_PATH] # Saved weights
```
You can check the reconstruction accuracy, by loading weights. 
They are located in either of `reg_model/*/model` folders.
```
python reconstruct.py --test ./data/merged/test.txt   \
                      --vocab ./data/merged/vocab.txt \
                      --hidden 612                    \
                      --depth 3                       \
                      --latent 256                    \
                      --model reg_model/merged_qdb9_256_612_homo/model
```
Parameters should match with model stored in "path_to_model" folder
Replace `test.txt` with `valid.txt` to test the validation accuracy (for hyperparameter tuning).
## Training Regression
```
regtrain.py [OPTIONS]
            --train [FILE_PATH] # Path to folder with smiles in txt format
            --train_prop [FILE_PATH] # Path to folder with properties corresponding to smiles in txt format
            --vocab [FILE_PATH] # Path to vocabulary used for JT decomposition
            --hidden [INT] # Size of the hidden state size of the model
            --depth [INT] # Depth of message passing
            --latent [INT] # Size of latent representation of molecule
            --batch [INT] # Batch size
            --lr [FLOAT] # Learning rate (during the last stage is set to 1e-3
            --modelvae [FILE_PATH] # Saved weights
            --save_dir [FILE_PATH] # Path to store weights after trainning
```
To follow our results, run
```
 python regtrain.py --train ./data/merged/reg_data/train_mols.txt      \
                    --train_prop ./data/merged/reg_data/train_homo.txt \
                    --vocab ./data/merged/vocab.txt                    \
                    --hidden 612                                       \
                    --depth 3                                          \
                    --latent 256                                       \
                    --batch 512                                        \
                    --lr 0.001                                         \
                    --modelvae vae_model/model.final                   \
                    --save_dir reg_model/
```
## Testing Regression
To test the regression you can use
```
regtrain.py [OPTIONS]
            --test [FILE_PATH] # Path to folder with smiles in txt format
            --test_prop [FILE_PATH] # Path to folder with properties corresponding to smiles in txt format
            --vocab [FILE_PATH] # Path to vocabulary used for JT decomposition
            --hidden [INT] # Size of the hidden state size of the model
            --depth [INT] # Depth of message passing
            --latent [INT] # Size of latent representation of molecule
            --batch [INT] # Batch size
            --lr [FLOAT] # Learning rate (during the last stage is set to 1e-3
            --modelvae [FILE_PATH] # Saved weights
            --modelreg [FILE_PATH] # Path to saved regerssion head weights
```
To look at the model described in our work, run
```
 python prop_test.py --test ./data/reg_data/test_mols.txt                \
                     --test_prop ./data/reg_data/test_lumo.txt           \
                     --vocab ./data/merged/vocab.txt                     \
                     --hidden 612                                        \
                     --depth 3                                           \
                     --latent 256                                        \
                     --batch 512                                         \
                     --lr 0.001                                          \
                     --modelvae reg_model/merged_qdb9_256_612_homo/model \
                     --modelvae reg_model/merged_qdb9_256_612_homo/model_regression
```

## Finetunning decoder
Note, that after regression training, the decoder will not work anymore. 
To make it reconstruct again it needs finetunning.
For that, you can use
```
vaetrain.py [OPTIONS]
            --train [FILE_PATH] # Path to folder with smiles in txt format
            --vocab [FILE_PATH] # Path to vocabulary used for JT decomposition
            --hidden [INT] # Size of the hidden state size of the model
            --depth [INT] # Depth of message passing
            --latent [INT] # Size of latent representation of molecule
            --batch [INT] # Batch size
            --lr [FLOAT] # Learning rate (during the last stage is set to 1e-3
            --model [FILE_PATH] # Saved weights
            --save_dir [FILE_PATH] # Path to store weights after trainning
```
To follow our work, run
```
dectrain.py --train ./data/merged/train.txt  \
            --vocab ./data/merged/vocab.txt  \
            --hidden 612                     \
            --depth 3                        \
            --latent 256                     \
            --batch 64                       \
            --lr 0.0001                      \
            --model ./reg_model/model        \
            --save_dir ./reg_model           \
```