
# Deep Latent-Variable Models i.e., Variational Autoencoders, for Dimensionality Reduction in Surrogate-Assisted Optimization (SAO) 

## Introduction
This Code repository contains the Programable Pythonic implementations of Variational Autoencoders [[1]](#1) for dimensionality reduction in SAO.


### Detail

#### 0 Data Generation
- Use Code in this [repository](https://github.com/SibghatUllah13/Deep-Latent_Variable_Models-for-dimensionality-reduction-in-surrogate-assisted-optimization/tree/master/Data%20Generation) to generate the data set for the surrogate-modeling. 

#### 1 VAEs Implementation

- Use Code in this [repository](https://github.com/SibghatUllah13/Deep-Latent_Variable_Models-for-dimensionality-reduction-in-surrogate-assisted-optimization/tree/master/VAES) to implement VAEs for each of the three different values for dimensioality.

#### 2 Hyper-Parameters Optimization

- The code for the Hyper-Parameters Optimization i.e., based on Grid-Search is provided [here](https://github.com/SibghatUllah13/Deep-Latent_Variable_Models-for-dimensionality-reduction-in-surrogate-assisted-optimization/tree/master/Hyper%20Parameters%20Optimization). Note that this code is common for both sets of surrogate-models i.e., baseline surrogates and the low-dimensional surrogate-models. Therefore, please utilize/customzie this code as per the need.

#### 3 Modeling Accuracy

- The soure code for the modeling accuracy for the baseline surrogates is [provided](https://github.com/SibghatUllah13/Deep-Latent_Variable_Models-for-dimensionality-reduction-in-surrogate-assisted-optimization/tree/master/Modeling%20Accuracy%20-%20Baseline%20Surrogates). 
- In addition, the source code for the low-dimensional surrogates is [provided](https://github.com/SibghatUllah13/Deep-Latent_Variable_Models-for-dimensionality-reduction-in-surrogate-assisted-optimization/tree/master/Modeling%20Accuracy%20-%20Latent%20Surrogates). Note that for the low-dimensional surrogates, it is required to first select the size of the dimensionality and then go to the corresponding sub-repo e.g., 50 % etc.

#### 4 Optimality

- The soure code for the Optimality i.e., by employing Cumulative Matrix Adaptive Evolution Strategies (CMA-Es) for the baselines surrogates is present [here](https://github.com/SibghatUllah13/Deep-Latent_Variable_Models-for-dimensionality-reduction-in-surrogate-assisted-optimization/tree/master/Original%20-%20Optimality). 
- In addition, the source code for the optimality regarding the low-dimensional surrogates is [provided](https://github.com/SibghatUllah13/Deep-Latent_Variable_Models-for-dimensionality-reduction-in-surrogate-assisted-optimization/tree/master/Latent%20-%20Optimality).


## Requiremnts
For this project to run you need:
* Python 3.7.3
* Pytorch 1.3.0+cpu
* Numpy 1.16.2
* Matplotlib
* Pandas 0.24.2
* Scikit-learn 0.20.3 

## Note

- It is very important to note that this is the minimum working code. Therefore, it is expected that at some stages of the implementation, one has to reuse part of the source code provided. For further questions on the code and the implementation, do not hesitate to contact: s.ullah@liacs.leidenuniv.nl. If you're further interested in our research, please visit our [official page](https://ecole-itn.eu/publications/).

## Acknowledgement

- This work has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement number 766186.

## References:
<a id="1">[1]</a> 
Kingma, D.P., Welling, M.: Auto-encoding variational bayes. In: 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings (2014).

