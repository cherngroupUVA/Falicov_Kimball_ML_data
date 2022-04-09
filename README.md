# Falicov_Kimball_ML_data
![e35f535ffd4f55f7e8b5a35f3e9db5d](https://user-images.githubusercontent.com/38637473/162581878-e6ee9a25-0f1d-4c91-8a57-7df29de2bb21.png)
## Introduction
This repository includes codes, trained model samples and data samples to successfully run the machine learning spin dynamics. This will reproduce results in the paper https://arxiv.org/abs/2105.13304. Sub-directories in this repo are:
1. *training_data_sample*:

      training data samples, from 30x30 lattice spin exact-diagonalization simulations.
            
2. *training*:

      this folder includes codes for training a only one possible jumping direction model. The error decreasing can be illustrated with the training data included in this folder. Go to this folder and do:
      
      ```shell
      python training.py
      ```
      you can see the prediction error decrease as the training goes on. The trained model at some points will be saved for further use.

      
3. *simulation*：

      in the root folder and do following, a trained model (this model is used to generate FIG. 2 and FIG. 3 in the paper) is already included in *model* folder and this code will do machine learning dynamics directly in a 150x150 lattice.

      ```shell
      python main.py filename
      ```

      the calculated f-electron configurations will to saved to root folder with name "config(filename).csv". Each row corresponds to empty or occupied sites in 1 dimension for a time step. 
      
 5. *data_example*:
      
      this folder include simulation result from machine learning or exact-diagonalization that is used to plot FIG. 1, FIG. 2, FIG.3 and FIG. 4 in the paper. 
      
When you reach this line, you have all you need to reproduce the results in https://arxiv.org/abs/2105.13304. If you have more questions, please contact sz8ea@virginia.edu for information.
