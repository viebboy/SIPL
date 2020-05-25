# Subset Sampling For Progressive Neural Network Learning 

Abstract: Progressive Neural Network Learning is a class of algorithms that incrementally construct the network's topology and optimize its parameters based on the training data. While this approach exempts the users from the manual task of designing and validating multiple network topologies, it often requires an enormous number of computations. In this paper, we propose to speed up this process by exploiting subsets of training data at each incremental training step. Three different sampling strategies for selecting the training samples according to different criteria are proposed and evaluated. We also propose to perform online hyperparameter selection during the network progression, which further reduces the overall training time. Experimental results in object, scene and face recognition problems demonstrate that the proposed approach speeds up the optimization procedure considerably while operating on par with the baseline approach exploiting the entire training set throughout the training process.


# Source Code
The code is written in Python 3.x with the following dependencies:
- dill 
- keras 2.2.4
- tensorflow 1.13.1
- scikit-learn 0.23.1
- joblib 0.15.1

# Data
The data should be downloaded from [this](https://tuni-my.sharepoint.com/:u:/r/personal/thanh_tran_tuni_fi/Documents/DONOTREMOVE/SIPL_data.tar.gz?csf=1&web=1&e=nmDf6X), and put under the main directory. Basically the directory structure should be like this:

``` 
SIPL/
│   README.md
└───data/
    │    caltech256_x_test.npy
    │ 	 caltech256_x_train.npy
    │	 ...   
└───code/
    │    exp_configurations.py
    │    ...
```

# Contact
Any inquiry can be sent to thanh.tran@tuni.fi or viebboy@gmail.com

# Citation
