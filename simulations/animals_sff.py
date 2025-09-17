import utils
import model_training as model

from animals import dataset

import numpy as np

import jax
import jax.numpy as jnp

import os
from importlib import reload
import GPUtil


############################################################
##                       GPU_MNGT                         ##

# Function to check and assign a free GPU
def assign_free_gpu():
    gpus = GPUtil.getAvailable(order='memory', limit=1)  # Get the GPU with the most free memory
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])  # Assign this GPU for the task
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4" # allow 40% of memory 
        os.environ["XLA_FLAGS"] = "--xla_gpu_mlir_emitter_level=4"
        print(f"Using GPU: {gpus[0]}")
    else:
        print("No free GPU found. Falling back to CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Use CPU if no GPUs are available

# Call the function to assign a GPU
assign_free_gpu()


############################################################
##                       DATASET                          ##

def rearange_labels_to_minimum_onehot(y, selected_classes):
    selected_classes = jnp.array(selected_classes)

    def map_to_new_index(idx):
        idx = jnp.array(idx).reshape(-1)  # Ensure idx is 1D
        match_matrix = selected_classes == idx[:, None]  # Shape: (N, num_selected)
        mapped = jnp.where(match_matrix, jnp.arange(len(selected_classes)), -1)
        return mapped.max(axis=1)

    return map_to_new_index(y)



os.environ["DATA"] = f"{os.getcwd()}"

nb_feat = 32
selected_classes = [0,1,2,3]
path = "animals/animals_features.h5"

chi2_test = False

dataset_params = dataset.AnimalsConfig(num_features = nb_feat, select_classes = selected_classes, chi2_test_on_feat_sel = chi2_test)


train_set = dataset.AnimalsDataset(config = dataset_params, train = True,  path = path)
selected_features = train_set.get_selected_feat() if chi2_test else None

test_set  = dataset.AnimalsDataset(config = dataset_params, train = False, path = path, sel_feat = selected_features)
val_set  = dataset.AnimalsDataset(config = dataset_params, val = True, path = path, sel_feat = selected_features)

train_set_x, train_set_y = train_set[:]
test_set_x, test_set_y = test_set[:]
val_set_x, val_set_y = val_set[:]

train_set_x = train_set_x.T
test_set_x  = test_set_x.T
val_set_x   = val_set_x.T

# reattribute a label depending on the number of classes if we extract a subset of classes
train_set_y = rearange_labels_to_minimum_onehot(train_set_y, selected_classes)
test_set_y  = rearange_labels_to_minimum_onehot(test_set_y, selected_classes)
val_set_y   = rearange_labels_to_minimum_onehot(val_set_y, selected_classes)

# # If true, the supervised training is used
supervised=True
label_strength = 1.0
# generate the positive and negative dataset in [Channel, Features] format
key = jax.random.PRNGKey(43)
key, subkey = jax.random.split(key)

# generate supervised or unsupervised dataset, and transpose in the good dimensions
train_set_x, train_set_pos_x, train_set_neg_x, train_set_y, test_set_x, test_set_y = utils.gen_dataset(train_set_x, train_set_y, test_set_x, test_set_y, supervised, subkey, label_strength)
_, _, _, _, val_set_x,  val_set_y  = utils.gen_dataset(train_set_x, train_set_y, val_set_x, val_set_y, supervised, subkey, label_strength)

print(f"Train set is of shape: {train_set_x.shape}")
print(f"Test set is of shape: {test_set_x.shape}")
print(f"Validation set is of shape: {val_set_x.shape}")

print(f"Train labels is of shape: {train_set_y.shape}")
print(f"Test labels is of shape: {test_set_y.shape}")
print(f"Validation labels is of shape: {val_set_y.shape}")

# # ############################################################
# # ##                      FF TRAINING                       ##

os.environ["DATA"] = f"{os.getcwd()}"
path = "jam/WeakResets/march_slope_x3_5k.hdf5"

# Jax random key
rngKey = jax.random.PRNGKey(32)


# Hyperparameters

batch_size = 16

memristor_layer = True # set to True to use OxRAM

nepoch1 = 10
nepoch2 = 20

scale_mem1 = 9500
scale_mem2 = 2000

scale_w1 = scale_mem1 if memristor_layer else 1/np.sqrt(32)
scale_w2 = scale_mem2 if memristor_layer else 1/np.sqrt(48)

scale1 = 0.0 if not memristor_layer else 0
scale2 = 0.0 if not memristor_layer else 0
 
theta1_p = 45.0
theta1_n = 3.0

theta2_p = 6.4
theta2_n = 2.2

thres1 = 1.1e-4 if memristor_layer else 0
thres2 = 3.2e-3 if memristor_layer else 0

# LogSigmoidSupervised is for SFF hidden layers
# Available learning methods
# - LogSigmoidSupervised : Supervised Forward Forward hidden layers (such as done in Hinton's paper)
# - LogSigmoid           : Competitive learning with clusters (can be used as classifier and hidden layers)
# - LogSigmoid_temp      : Competitive learning with clusters (can be used as classifier and hidden layers) with temperature coefficient, eta = 1
# - LogSigmoidHidden     : Competitive learning with clusters (can be used as classifier and hidden layers) with temperature coefficient, eta = -1
# - SoftMax              : SoftMax classifier

act1 = "LogSigmoidSupervised" 
act2 = "LogSigmoid_temp"

# Update layers' configurations
LayersConfigFF1 = (nepoch1, scale1, theta1_p, theta1_n, memristor_layer, scale_w1, thres1, act1, path)
LayersConfigFF2 = (nepoch2, scale2, theta2_p, theta2_n, memristor_layer, scale_w2, thres2, act2, path)

print(f"Training is in {'OxRAM' if memristor_layer else 'FP32'}")

print(f"scale_mem1 is {scale_mem1}")
print(f"scale_mem2 is {scale_mem2}")

print(f"act1 is {act1}")
print(f"act2 is {act2}")

print(f"nepoch1 = {nepoch1}, scale1 = {scale1:.5f}, theta1_p = {theta1_p:.4f}, theta1_n = {theta1_n:.4f}, thres1 = {thres1}")
print(f"nepoch2 = {nepoch2}, scale2 = {scale2:.5f}, theta2_p = {theta2_p:.4f}, theta2_n = {theta2_n:.4f}, thres2 = {thres2}")

jax.clear_caches()
reload(model)

nb_run = 100

test_acc_sum = 0.
train_acc_sum = 0.

test_acc_mean = [0, 0, 0, 0, 0, 0, 0, 0]
train_acc_mean = [0, 0, 0, 0, 0, 0, 0, 0]

test_acc_hist = []
train_acc_hist = []

loss_hists = []

pos_pulses_hist = []
neg_pulses_hist = []

for i in range(nb_run):

    # shuffling the dataset
    permutations = jax.random.permutation(jax.random.split(rngKey)[1], train_set_x.shape[0])

    train_set_x = train_set_x[permutations]
    train_set_y = train_set_y[permutations]

    jax.clear_caches()
    reload(model)

    rngKey = jax.random.split(rngKey)[1]
    subkey = jax.random.split(rngKey, 6)


    # Declare each layer with input neurons, output neurons, hyperparameters, random_key
    LayersFF = [
        model.LayerParams(36, 48, *LayersConfigFF1, subkey[1]),
        model.LayerParams(48, 48, *LayersConfigFF2, subkey[2]),
    ]
    
    # Call training function
    TrainedLayersFF, test_acc, train_acc, loss_hist, pos_pulses, neg_pulses = model.ForwardModel.train(LayersFF, batch_size,
                            train_set_x, train_set_y,
                            test_set_x, test_set_y,
                            supervised = supervised,
                            classifier_on_all_act = False,
                            acc_each_epoch = True)
    
    test_acc_sum += test_acc[-1].mean()
    train_acc_sum += train_acc[-1].mean()

    test_acc_hist.append(jnp.array(test_acc))
    train_acc_hist.append(jnp.array(train_acc))


    pos_pulses_hist.append(pos_pulses)
    neg_pulses_hist.append(neg_pulses)

    loss_hists.append(jnp.array(loss_hist))

    epochs_acc = 0
    for l in range(len(LayersFF)):
        epochs_acc += LayersFF[l].Epochs
        test_acc_mean[l] += test_acc[epochs_acc-1].mean()
        train_acc_mean[l] += train_acc[epochs_acc-1].mean()
        print(f"Average test accuracy on layer {l} after run {i} is {test_acc_mean[l] / (i+1) * 100}%")
        print(f"Average train accuracy on layer {l} after run {i} is {train_acc_mean[l] / (i+1) * 100}%\n")

run_parameters = {
    "experiment_settings": {
        "path": path,
        "n_runs": nb_run,
        "batch_size": batch_size,
        "supervised": supervised
    },
    "LayersFF": [
        {
            f"Layer {layer_idx}": utils.serialize(vars(layer))
        }
        for layer_idx, layer in enumerate(LayersFF)
    ]
}

# logs
utils.save_logs(run_parameters, test_acc_hist, train_acc_hist, loss_hists, pos_pulses_hist, neg_pulses_hist, title = "ANIMALS Accuracy on BackProp MLP", plot=False)
    

# final accuracy over all runs
print(f"\nAverage train accuracy after run {i} is {train_acc_sum / (i+1) * 100}%")
print(f"Average test accuracy after run {i} is {test_acc_sum / (i+1) * 100}% \n")