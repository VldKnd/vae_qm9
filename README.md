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
* Python (version == 2.7)
* PyTorch (version >= 0.2)
```
# Structure:
```
.
├── jtnn                ### Implementation of JT VAE
│   └── **              ### For further detail consult the original github https://github.com/wengong-jin/icml18-jtnn.
├── fast_jtnn           ### Speeded up version
│   └── ** 
├── qvae
│   ├── model.py        ### Custom implementation of ResNet blocks and architecture.            
│   ├── qmodel.py       ### Quantized blocks.       
│   ├── train.py        ### Function for training and validation of neural network.
│   ├── utils.py        ### Utilities.
│   └── __init__.py
└── README.md
```
# Contact
Vladimir Kondratyev (kondratyev.w.i@gmail.com)