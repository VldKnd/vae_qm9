# Generative modeling based on junction tree variational autoencoder for HOMO value prediction and molecular optimization
In this work, we provide further development of the junction tree variational
autoencoder (JT VAE) architecture in terms of implementation and application
of the internal feature space of the model. Pretraining of JT VAE on a large
dataset and further optimization with a regression model led to a latent space
that can solve several tasks simultaneously: prediction, generation, and
optimization. We use the ZINC database as a source of molecules for the JT VAE
pretraining and the QM9 dataset with its HOMO values to show the application
case. We evaluate our model on multiple tasks such as property (value)
prediction, generation of new molecules with predefined properties, and structure
modification toward the property. Across these tasks, our model shows
improvements in generation and optimization tasks while preserving the precision
of state-of-the-art models

The Official implementation of our [Junction Tree Variational Autoencoder](https://arxiv.org/abs/1802.04364)
is taken from `https://github.com/wengong-jin/icml18-jtnn` page.
## Reconstruction accuracy of auto encoders
We give the results of a basic JT-VAE training. The columns Train
and Test correspond to the respective datasets used in the process. The column
Acc contains the percentage of accurately reconstructed molecules by the encoder
decoder pair. The column KL indicates whether the Kullback-Leibler divergence
penalty term was used in the second stage of the training.

| Train | Test | Accuracy | KL Divergence       |
|-------|------|----------|---------------------|
| MIX   | QM9  | 83.1%    | True                |
| QM9   | QM9  | 81.9%    | True                |
| QM9   | QM9  | 79.4%    | False               |
| MIX   | QM9  | 81%      | False               |
| ZINC  | QM9  | 75%      | True                |
## Regression
The lines Ridge Regr. and Elastic Net correspond to the training of the eponymous regression from latent space into the
property of interest (HOMO). The last two lines correspond to the training of the
unfrozen encoder (pretrained during the basic JT VAE training) jointly with two layers feed forward neural network with ReLu activations. We measure the HOMO accuracy by MAE loss

| Method        | Train| Test | HOMO MAE |
|---------------|------|------|----------|
| Ridge Regr.   | QM9  | QM9  | 0.18     |
| Elastic Net   | QM9  | QM9  | 0.34     |
| JT-ENC + FNNN | QM9  | QM9  | 0.09     |
| JT-ENC + FNNN | MIX  | QM9  | 0.09     |

# Requirements
```
* Linux (We only tested on Ubuntu)
* RDKit (version >= 2017.09)
* Python (version >= 3.8)
```
# Structure:
```
.
├── jtnn                   ### Implementation of JT VAE
│   └── **                 ### For further detail consult the original github https://github.com/wengong-jin/icml18-jtnn.
│
├── fast_jtnn              ### Speeded up version
│   └── ** 
│ 
├── qvae ## For More information go to qvae/README.md
│   │ 
│   ├── data
│   │     ├── merged       ### Folder with mixed QM9 and ZINC smiles.
│   │     └── reg_data     ### Folder with QM9 smiles and its homo/lumo/gap values.
│   │ 
│   ├── enc_model          ### Empty folder made for conviniance of training.
│   ├── var_model          ### Empty folder made for conviniance of training.
│   ├── reg_model          ### Folder with pre-train weights for the model.
│   │     └── merged_*     ### Different pretrained models
│   │ 
│   ├── mol_gen            ### Folder with scripts to generate molecules with gradient descent.
│   │     ├── ctv.py       ### Closest by target value to MIX dataset generation.
│   │     ├── gauss.py     ### Generation from random vectors.
│   │     └── funcs.py     ### Additional functions.
│   │ 
│   ├── pretrain.py        ### Script for training Auto-Encoder JT-VAE without KL divergence penalty.
│   ├── vaetrain.py        ### Script for VAE training.
│   ├── regtrain.py        ### Script for training the regression head.
│   ├── dectrain.py        ### Script for finetunning the decoder of the model.
│   │ 
│   ├── reconstruct.py     ### Script for testing the reconstruction accuracy of the model. 
│   └── prop_test.py       ### Script for testing the MAE accuracy of the regressor.
│          
└── README.md
```
# Contact
Vladimir Kondratyev (kondratyev.w.i@gmail.com)