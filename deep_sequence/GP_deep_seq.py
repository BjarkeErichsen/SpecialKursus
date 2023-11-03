"""
ENVIRONMENT = deep_sequence
In this file we train the GP model.
Depends on having
1 deep_sequence downloaded at /home/bjarke/Desktop/MLDEcode/deep_sequence/
2 MSA downloaded at /home/bjarke/Desktop/MLDEcode/deep_sequence/DeepSequence/examples/datasets/
3 parameters being saved at locally at -arams/.."
4 Set file prefix to the base name (excluding the _params/_v) of the parameter files
TRAIN:
    1 Have a csv file of training sequences
    2 Use deep sequence to embed the training sequences
    3 Use a GP to model a function inside the latent space representing of the fitness
"""
import theano
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/bjarke/Desktop/MLDEcode/deep_sequence/DeepSequence/DeepSequence/")
sys.path.insert(0, "/home/bjarke/Desktop/MLDEcode/deep_sequence/DeepSequence/DeepSequence/examples")
import model
import helper
import train_DS
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt


data_params = {"dataset": "Gb1"}
data_helper = helper.DataHelper(
                dataset=data_params["dataset"],
                working_dir="/home/bjarke/Desktop/MLDEcode/deep_sequence/DeepSequence/examples",
                calc_weights=False
                )

model_params = {
    "bs"                :   100,
    "encode_dim_zero"   :   1500,
    "encode_dim_one"    :   1500,
    "decode_dim_zero"   :   100,
    "decode_dim_one"    :   500,
    "n_latent"          :   30,
    "logit_p"           :   0.001,
    "sparsity"          :   "logit",
    "f_nonlin"          :  "sigmoid",
    "fps"               :   True,
    "n_pat"             :   4,
    "r_seed"            :   1,
    "conv_pat"          :   True,
    "d_c_size"          :   40,
    "sparsity_l"        :   1.0,
    "l2_l"              :   1.0,
    "dropout"           :   True,
    }

data_helper = helper.DataHelper(dataset=data_params["dataset"], calc_weights=True)

vae_model = model.VariationalAutoencoderMLE(data_helper,
                                            batch_size=model_params["bs"],
                                            encoder_architecture=[model_params["encode_dim_zero"],
                                                                  model_params["encode_dim_one"]],
                                            decoder_architecture=[model_params["decode_dim_zero"],
                                                                  model_params["decode_dim_one"]],
                                            n_latent=model_params["n_latent"],
                                            logit_p=model_params["logit_p"],
                                            encode_nonlinearity_type="relu",
                                            decode_nonlinearity_type="relu",
                                            final_decode_nonlinearity=model_params["f_nonlin"],
                                            final_pwm_scale=model_params["fps"],
                                            conv_decoder_size=model_params["d_c_size"],
                                            convolve_patterns=model_params["conv_pat"],
                                            n_patterns=model_params["n_pat"],
                                            random_seed=model_params["r_seed"],
                                            sparsity_lambda=model_params["sparsity_l"],
                                            l2_lambda=model_params["l2_l"],
                                            sparsity=model_params["sparsity"])


"""
Standard encoding and decoding of elements in the MSA
"""
file_prefix = "vae_output_encoder-1500-1500_Nlatent-30_decoder-100-500_dataset-Gb1_bs-100_conv_pat-True_d_c_size-40_dropout-True_f_nonlin-sigmoid_fps-True_l2_l-1.0_logit_p-0.001_n_pat-4_r_seed-1_sparsity-logit_sparsity_l-1.0"
vae_model.load_parameters(file_prefix=file_prefix)
focus_seq_one_hot = np.expand_dims(data_helper.one_hot_3D(data_helper.focus_seq_trimmed),axis=0)
mu_blat, log_sigma_blat = vae_model.recognize(focus_seq_one_hot)
z_blat = vae_model.encode(focus_seq_one_hot)

"""
Plots and other extra features
"""
def plot_prop_dist(z_blat):
    seq_reconstruct = vae_model.decode(z_blat)
    plt.figure(figsize=(35,10))
    plt.imshow(seq_reconstruct[0].T,cmap=plt.get_cmap("Blues"))
    ax = plt.gca()
    ax.set_yticks(np.arange(len(data_helper.alphabet)))
    ax.set_yticklabels(list(data_helper.alphabet))
    plt.show()





##############################  Gausian Process training ########################
"""
Training the GP on datapoints in the .csv
The following needs to be defined:
1. Batch size 
2. Path towards csv file  
I am VERY sure the translation of the AA's to one hot matches the format in the MSA/deep sequence train set. 
"""
# Example of using the function
filepath = '/home/bjarke/Desktop/Data/DMS/project/train_test_splits/train_set.csv'
batch_size = 5  #the number of datapoints we encode at a time
def amino_acid_to_one_hot(amino_acid, amino_acid_dict):
    one_hot = np.zeros(len(amino_acid_dict))
    one_hot[amino_acid_dict[amino_acid]] = 1
    return one_hot

def sequence_to_one_hot(sequence, amino_acid_dict):
    return np.array([amino_acid_to_one_hot(aa, amino_acid_dict) for aa in sequence])

def load_data_in_batches(filepath, batch_size=1):
    # Define your amino acids alphabet, ensure it includes all possible amino acids in your sequences
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    amino_acid_dict = {aa: idx for idx, aa in enumerate(amino_acids)}

    data = pd.read_csv(filepath)
    n_sequences = len(data)
    n_batches = (n_sequences + batch_size - 1) // batch_size

    for i in range(n_batches):
        batch_sequences = data['Variants'][i*batch_size:(i+1)*batch_size]
        batch_fitness = data['Fitness'][i*batch_size:(i+1)*batch_size].values

        one_hot_batch = np.array([sequence_to_one_hot(seq, amino_acid_dict) for seq in batch_sequences])

        yield one_hot_batch, batch_fitness



# Instantiate a Gaussian Process model
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
gp = GaussianProcessRegressor(kernel=kernel)

# Prepare data for Gaussian Process
X_train = []
y_train = []

k = 0
for one_hot_sequences, fitness in load_data_in_batches(filepath, batch_size=batch_size):
    z = vae_model.encode(one_hot_sequences)
    X_train.append(z)
    y_train.extend(fitness)
    k += batch_size
    if 100 < k:
        print("We train the GP on the first ", k, " of the datapoints for computational reasons")
        break
X_train = np.concatenate(X_train, axis=0)
y_train = np.array(y_train)

gp.fit(X_train, y_train)

y_pred, sigma = gp.predict(X_train, return_std=True)

plt.figure(figsize=(12, 6))
plt.scatter(np.arange(len(y_train)), y_train, color='k', label='True values')
plt.plot(y_pred, color='b', label='GP mean predictions')
plt.fill_between(np.arange(len(y_train)), y_pred - sigma, y_pred + sigma, color='b', alpha=0.2, label='Confidence interval')
plt.title('Gaussian Process Regression')
plt.xlabel('Data Point Index')
plt.ylabel('Fitness')
plt.legend()
plt.show()


###########################################################################         TESTING        ##############################################################################################################
"""
ENVIRONMENT = deep_sequence
In this file we use the trained deep sequence and GP model to do the following.
We will run 2 types of tests

TEST 1: Discriminative modelling
    1 Have a set of test Gb1 sequences with associated fitness values
    2 For each sequence embed the given sequence
    3 Predict the fitness of the given sequence using the GP model (as well as the variance in the prediction)
    4 Compare with target values for fitness

TEST 2:  Generative modelling
    1 Have a set of test Gb1 sequences with associated fitness values
    2 Sample from gaussian process using a preselected aquisition function
    3 Decode said sequence back to AA space
    4 Use the Gb1 test set as a lookup table (note that its not always the case that the Gb1 test dataset will contain the decoded AA sequence)
"""






