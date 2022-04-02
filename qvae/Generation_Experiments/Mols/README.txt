__Files_Structure__

-mol_n
"This folder containtes molecules, that have been found in a Nature(*) paper. Each folder corresponds to specific method of sampling molecules and looking for the optimised structure. The short methods description can be found lower."

    -1
       "Those folders contain the result of the experiments for specific molecules, which is saved in file **main_n.pdf**, if the molecule have been succesfully reconstructed one of the files will be marked with exctra star, e.g. fig_10*.pdf"
       
        -mol_0
            -fig_0.pdf
            ..
            -fig_k.pdf
            ..
            -main_0.pdf
        ..
        -mol_11
            ..
    -2
        ..
    -3
        ..
    -4
        ..
    -5
        ..
    -6
        ..
   
-mol_m
"This folder containtes carifully selected molecules, that have been reconstructed during the validation procedure. All other files follow the structure of files from , but with smaller (4) amount of molecules mol_n"

    -1       
        -mol_0
            -fig_0.pdf
            ..
            -fig_k.pdf
            ..
            -main_0.pdf
        ..
        -mol_4
            -..
    -2
        ..
    -3
        ..
    -4
        ..
    -5
        ..
    -6
        ..
    
    
    
    
*-maybe good thing to add paper here?


Methods
-1 Sampling the random vector from Normal Gaussian dist and then optimizing with gradient descent in latent space

-2 Taking the closest by target propertie(s) vector from train dataset, and adding to it the noise sampled the random vector from Normal Gaussian dist and then optimizing with gradient descent in latent space

-3 Computing the embedings of target molecule, taking the closest by distance from computed embedding vector from train dataset, and adding to it the noise sampled the random vector from Normal Gaussian dist and then optimizing with gradient descent in latent space

-4 The same initialization as in 1, but optimising the structure in double step system. First step consist in latent descent optimisation, second step consists in selecting top 10 molecules with closest to target propertie(s). The value of molecules properties is computed using neural network.

-5 The same initialization as in 2, but optimising the structure in double step system. First step consist in latent descent optimisation, second step consists in selecting top 10 molecules with closest to target propertie(s). The value of molecules properties is computed using neural network.

-6 The same initialization as in 2, but optimising the structure in double step system. First step consist in latent descent optimisation, second step consists in selecting top 10 molecules with closest to target propertie(s). The value of molecules properties is computed using neural network.
