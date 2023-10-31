This directory contains the code to simulate the results of 3 different models on a Deep mutational scan.

The 3 models are:
1. ESM embeddings + ensemble of classifiers   (Discriminative)
2. deep sequence embedding + gaussian process (Generative)
3. EvoDiff                                    (Generative)

To fairly compare these models we organize the following experiment.
1. Select parameter m, corresponding to the training dataset. This should match as closely as possible the screening capacity as the data here would be the previous rounds screened data.
2. Select parameter k, corresponding to the produced library. This should match the screening capacity of the next round. 
3. For discriminative models pick parameter n > k which is the number of sequences we will evaluate. We evaluate the model based on the max fitness achieved in the top k sequences chosen to be part of the next screening round. 
4. For generative models generate k sequences. Then use the lookup table to find the max fitness achieved.

The dataset used is:
1. GB1
- ![img.png](img.png)

The GB1 dataset as well as other related files reside under:
- /home/bjarke/Desktop/Data/DMS/

