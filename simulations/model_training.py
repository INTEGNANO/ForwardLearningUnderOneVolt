import functools
import os

from jam import jam

import jax
import jax.numpy as jnp
import numpy as np

from typing import List

from tqdm.auto import tqdm 

import warnings




"""
General purposes functions
"""

@functools.partial(jax.jit, static_argnames=["batch_size"])
def batch(arr, batch_size):
    indices = jnp.arange(0, len(arr), batch_size)
    
    def get_batch(i):
        if arr.ndim == 1:
            return jax.lax.dynamic_slice(arr, (i,), (batch_size,))
        else:
            return jax.lax.dynamic_slice(arr, (i, 0), (batch_size, arr.shape[1]))
    return jax.lax.map(get_batch, indices)


class LayerParams:
    
    def __init__(self, in_dim, out_dim, 
                 NbEpochs, 
                 LearningRate, 
                 PosThetaThreshold, 
                 NegThetaThreshold, 
                 MemristorWeights=False, 
                 ParamsScale=0.1, 
                 ThresholdGradient=1e-6, 
                 LossFunction="LogSigmoid", 
                 MemristorPath="WeakResets/march_slope_x3_5k.hdf5", 
                 rngKey=0):
            
            
            # initialize the fields
            if MemristorWeights == False :
                weights_seed, bias_seed = jax.random.split(rngKey)
                self.Params = {
                    "W" : jax.random.normal(weights_seed, (out_dim, in_dim), jnp.float32) * ParamsScale,
                    "b" : jnp.zeros((out_dim, ), jnp.float32) # unused
                }

            else:
                memristor_path = os.path.join(os.environ["DATA"], MemristorPath)
                
                memristor_params = {
                # Also, "diff", or "log", or function (ResistancePositive, ResistanceNegative) -> WeightValue
                "weight_formula": "conductance_diff",
                "scale_factor": ParamsScale,
                }

                rngKey_pool, *rngKey_pool = jax.random.split(rngKey, 6)
                self.Pool = jam.DevicePool(
                    memristor_path,
                    rngKey_pool[0],
                )

                self.Params = {
                    "W": jam.MemristorWeight.init(self.Pool, "W", (out_dim, in_dim), **memristor_params),
                    "b": jam.MemristorWeight.init(self.Pool, "b", (out_dim,), **memristor_params)
                }

            self.Epochs             = NbEpochs
            self.LearningRate       = LearningRate
            self.PosThetaThreshold  = PosThetaThreshold
            self.NegThetaThreshold  = NegThetaThreshold
            self.MemristorWeights   = MemristorWeights
            self.ParamsScale        = ParamsScale
            self.ThresholdGradient  = ThresholdGradient
            self.LossFunction       = LossFunction
            self._key               = rngKey

    def update_params(self, new_params, mask = None):
      if mask == None :
          self.Params = jax.tree_util.tree_map(lambda p_new : p_new, new_params) 
      else :
          self.Params = jam.map_params(jax.tree_util.tree_map(lambda m, p_new, p : jnp.where(m, p_new, p), mask, new_params, self.Params))


"""
Forward algorithm class
"""
class ForwardModel:

    @staticmethod
    def normalize(x: jnp.array):
        return x
        # return x / (jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True)) + 1e-6)



    @staticmethod
    @functools.partial(jax.vmap, in_axes=(0, None))
    @functools.partial(jax.jit, static_argnames=["activation_fn"])
    def forward_one_pass(x: jnp.ndarray, params: dict, activation_fn=jax.nn.relu) -> jnp.ndarray:
        """Perform forward pass with optional activation function."""
        x = ForwardModel.normalize(x)
        return activation_fn(params["W"] @ x)

    @functools.partial(jax.vmap, in_axes=(0, None))
    @functools.partial(jax.jit)
    def forward_with_softmax_one_pass(x: jnp.ndarray, params):
        x = ForwardModel.normalize(x)
        return jax.nn.softmax(params["W"] @ x)


    @functools.partial(jax.vmap, in_axes=(0, None, None))
    @functools.partial(jax.jit, static_argnames=['idx_last_layer', 'activation_last_layer'])
    def forward_with_softmax_all_activations(x, weights, idx_last_layer, activation_last_layer=jax.nn.softmax):
        weights = jam.eval_params(weights)

        last_layer_feat = None

        for idx in range(idx_last_layer+1):
            x = ForwardModel.normalize(x)
            if idx == idx_last_layer:
                # last layer
                last_layer_feat = x if last_layer_feat is None else last_layer_feat
                x_all_act = activation_last_layer(weights[f"W{idx}"] @ last_layer_feat)
                x = x
            else:
                x =  jax.nn.relu(weights[f"W{idx}"] @ x)
                last_layer_feat = x if last_layer_feat is None else jnp.concat([last_layer_feat, x], axis = -1)


        return x_all_act, x 

    @functools.partial(jax.vmap, in_axes=(0, None, None))
    @functools.partial(jax.jit, static_argnames=['idx_last_layer', 'activation_last_layer'])
    def forward_with_softmax(x, weights, idx_last_layer, activation_last_layer=jax.nn.softmax):
        weights = jam.eval_params(weights)

        for idx in range(idx_last_layer+1):
            x = ForwardModel.normalize(x)
            pre_act = weights[f"W{idx}"] @ x
            if idx == idx_last_layer:
                # last layer
                x = activation_last_layer(pre_act)
            else:
                x = jax.nn.relu(pre_act)


        return x, pre_act
    
    @functools.partial(jax.vmap, in_axes=(0, None, None))
    @functools.partial(jax.jit, static_argnames=['idx_last_layer'])
    def forward(x, weights, idx_last_layer):
        weights = jam.eval_params(weights)

        for idx in range(idx_last_layer+1):
            x = ForwardModel.normalize(x)
            pre_act = weights[f"W{idx}"] @ x
            x = jax.nn.relu(pre_act)


        return x, pre_act
    







    @staticmethod
    @functools.partial(jax.jit, static_argnames=['loss'])
    def loss_softmax(act, labels_oh, loss : str = "CrossEntropy"):
        if loss == "CrossEntropy":
            loss = -jnp.mean(jnp.sum(labels_oh * jnp.log(act + 1e-10), axis=-1)) # cross entropy loss
        else:
            loss = jnp.mean((act - labels_oh) ** 2) # MSE loss
        return loss
    
    @staticmethod
    @functools.partial(jax.jit, static_argnames=["theta_p", "theta_n"])
    def loss_fn_clusters_hidden(act, theta_p, theta_n, labels_oh):
        """
        Compute the loss and manual gradients for cluster-based learning,
        where all clusters are pulled up, positive pushed down.
        """

        batch_size, num_neurons = act.shape
        num_classes = labels_oh.shape[-1]
        neurons_per_cluster = num_neurons // num_classes

        # Cluster polarity mask (True for activations with cluster index matching the class)
        polarity = jnp.kron(labels_oh, jnp.ones((1, neurons_per_cluster)))

        polarity_p = (1-polarity)  * jnp.power(act, 2)
        polarity_n =    polarity   * jnp.power(act, 2)

        g_p = theta_p *  polarity_p.sum(axis = - 1)
        g_n = theta_n * 1/(num_classes-1) * polarity_n.sum(axis = - 1) 

        return -jax.nn.log_sigmoid(jnp.concatenate([g_p, -g_n])).mean()
    
    @staticmethod
    @functools.partial(jax.jit, static_argnames=["theta_p", "theta_n"])
    def loss_fn_clusters(act, theta_p, theta_n, labels_oh):
        """
        Compute the loss and manual gradients for cluster-based learning,
        where positive clusters are pulled up, negatives pushed down.
        """

        batch_size, num_neurons = act.shape
        num_classes = labels_oh.shape[-1]
        neurons_per_cluster = num_neurons // num_classes

        # Cluster polarity mask (True for activations with cluster index matching the class)
        polarity = jnp.kron(labels_oh, jnp.ones((1, neurons_per_cluster)))

        polarity_p =    polarity    * jnp.power(act, 2)
        polarity_n = (1-polarity)   * jnp.power(act, 2)

        g_p =                     polarity_p.sum(axis = - 1) - theta_p
        g_n = 1/(num_classes-1) * polarity_n.sum(axis = - 1) - theta_n

        return -jax.nn.log_sigmoid(jnp.concatenate([g_p, -g_n])).mean()

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["theta_p", "theta_n"])
    def loss_fn_clusters_temp(act, theta_p, theta_n, labels_oh):
        """
        Compute the loss and manual gradients for cluster-based learning,
        where positive clusters are pulled up, negatives pushed down.
        """

        batch_size, num_neurons = act.shape
        num_classes = labels_oh.shape[-1]
        neurons_per_cluster = num_neurons // num_classes

        # Cluster polarity mask (True for activations with cluster index matching the class)
        polarity = jnp.kron(labels_oh, jnp.ones((1, neurons_per_cluster)))

        polarity_p =    polarity    * jnp.power(act, 2)
        polarity_n = (1-polarity)   * jnp.power(act, 2)

        g_p = theta_p *  polarity_p.sum(axis = - 1)
        g_n = theta_n * 1/(num_classes-1) * polarity_n.sum(axis = - 1) 

        return -jax.nn.log_sigmoid(jnp.concatenate([g_p, -g_n])).mean()


    @staticmethod
    @functools.partial(jax.jit, static_argnames=["theta_p", "theta_n"])
    def loss_fn_supervised(act_p, act_n, theta_p, theta_n):
        
        g_p =  jnp.power(act_p,2).sum(axis = - 1) - theta_p
        g_n =  jnp.power(act_n,2).sum(axis = - 1) - theta_n

        return -jax.nn.log_sigmoid(jnp.concatenate([g_p, -g_n])).mean()







    @functools.partial(jax.jit)
    def generate_neg_batch_random(x, labels_oh):
                    
        nb_classes = labels_oh.shape[-1]

        # true random (slower, no bias)
        # rand_labels_choice = jnp.repeat(jnp.arange(nb_classes), labels_oh.shape[0], axis = 0)
        # rand_labels_choice = jnp.delete(rand_labels_choice, jnp.argmax(labels_oh, axis = -1), assume_unique_indices=True)

        # rand_idx = np.random.randint(nb_classes-1, size = x.shape[0])

        # rand_labels = rand_labels_choice[rand_idx]
        # rand_labels = jnp.eye(nb_classes)[rand_labels.astype(jnp.int32)]



        # random labels embedded (faster but slight bias)
        rand_labels = np.random.randint(nb_classes, size = x.shape[0])
        rand_labels = jnp.where(jnp.argmax(labels_oh, axis = 1) == rand_labels, (rand_labels + 1) % nb_classes, rand_labels)
        rand_labels = jnp.eye(nb_classes)[rand_labels.astype(jnp.int32)]

        return x.at[:, :nb_classes].set(rand_labels)

    @functools.partial(jax.jit, static_argnames=["label_strength"])
    def generate_pos_batch(x, labels_oh, label_strength = 1.0):
        nb_classes = labels_oh.shape[-1]
        labels_oh = label_strength * labels_oh
        return x.at[:,:nb_classes].set(labels_oh)
    
    @functools.partial(jax.jit, static_argnames=["label_strength"])
    def generate_neutral_batch(x, labels_oh, label_strength = 1.0):
        nb_classes = labels_oh.shape[-1]
        neutral_strength = label_strength / nb_classes
        return x.at[:,:nb_classes].set(neutral_strength)
    






    
    @functools.partial(jax.value_and_grad, argnums=1) 
    @functools.partial(jax.jit, static_argnames=['theta_p', 'theta_n', 'loss_function', 'idx_layer_grad', 'supervised', 'classifier_on_all_act'])
    def compute_grad(x, weights, theta_p, theta_n, loss_function, targets_onehot, idx_layer_grad, supervised, classifier_on_all_act): 

        if supervised:
            if loss_function == "LogSigmoidSupervised":
                x_p = ForwardModel.generate_pos_batch(x, targets_onehot)
                x_n = ForwardModel.generate_neg_batch_random(x, targets_onehot)
            else:
                x_p = ForwardModel.generate_neutral_batch(x, targets_onehot)
                x_n = jnp.zeros_like(x_p)
        else:
            x_p = x
            x_n = x

        if loss_function == "LogSigmoidSupervised":
            act_p, _ = ForwardModel.forward(x_p, weights, idx_layer_grad)
            act_n, _ = ForwardModel.forward(x_n, weights, idx_layer_grad)
        elif loss_function == "SoftMax":
            if classifier_on_all_act:
                act_p, _ = ForwardModel.forward_with_softmax_all_activations(x_p, weights, idx_layer_grad)
            else :
                act_p, _ = ForwardModel.forward_with_softmax(x_p, weights, idx_layer_grad)
            act_n = jnp.zeros_like(act_p)
        else:
            act_p, _ = ForwardModel.forward(x_p, weights, idx_layer_grad)
            act_n = jnp.zeros_like(act_p)


        if loss_function == "LogSigmoidSupervised":
            return ForwardModel.loss_fn_supervised(act_p, act_n, theta_p, theta_n)
        elif loss_function == "LogSigmoidHidden":
            return ForwardModel.loss_fn_clusters_hidden(act_p, theta_p, theta_n, targets_onehot)
        elif loss_function == "LogSigmoid":
            return ForwardModel.loss_fn_clusters(act_p, theta_p, theta_n, targets_onehot)
        elif loss_function == "LogSigmoid_temp":
            return ForwardModel.loss_fn_clusters_temp(act_p, theta_p, theta_n, targets_onehot)
        elif loss_function == "SoftMax":
            return ForwardModel.loss_softmax(act_p, targets_onehot)
        




                       
    @functools.partial(jax.jit, static_argnames=['batch_size', 'nb_layers', 'learning_rate', 'grad_threshold', 'layer_to_update', 'theta_p', 'theta_n', 'loss_function', 'supervised', 'classifier_on_all_act'])
    def step(imgs, targets_onehot, params, theta_p, theta_n, loss_function, batch_size, nb_layers, learning_rate, grad_threshold, layer_to_update: int = -1,  supervised = False, classifier_on_all_act = False):

        @functools.partial(jax.jit, static_argnames=['nb_layers', 'batch_size', 'grad_threshold', 'layer_to_update', 'theta_p', 'theta_n', 'loss_function', 'supervised', 'classifier_on_all_act'])
        def update(imgs, targets_onehot, params, theta_p, theta_n, loss_function, batch_size, nb_layers, grad_threshold, layer_to_update : int = -1, supervised = False, classifier_on_all_act = False):

            # used specifically for the last batch, in case the train size is not multiple of the batch size
            batch_size_factor =  imgs.shape[0] / batch_size
            loss, grads = ForwardModel.compute_grad(imgs, jam.eval_params(params), theta_p, theta_n, loss_function, targets_onehot, layer_to_update, supervised, classifier_on_all_act)

            for grad_idx in range(nb_layers): 
                grads[f"W{grad_idx}"] = jnp.zeros(grads[f"W{grad_idx}"].shape) if grad_idx != layer_to_update and layer_to_update != -1 else grads[f"W{grad_idx}"]
                grads[f"b{grad_idx}"] = jnp.zeros(grads[f"b{grad_idx}"].shape) if grad_idx != layer_to_update and layer_to_update != -1 else grads[f"b{grad_idx}"]


            def weight_update(weight, grad):
                if isinstance(weight, jam.MemristorWeight):
                    return weight.incr_all(grad < -grad_threshold).decr_all(grad > grad_threshold)
                else:
                    # SignSGD in FP32 (unused, but can be uncommented to use)
                    # grads = lambda: jax.tree_util.tree_map(lambda gp : jnp.where(gp < -grad_threshold, -1, 0) + jnp.where(gp > grad_threshold, 1, 0), grad)
                    # grad = grads()

                    return jax.tree_util.tree_map(lambda a, b: a - learning_rate * batch_size_factor * b, weight, grad)

            return loss, jam.map_params(weight_update, params, grads)
        
        return update(imgs, targets_onehot, params, theta_p, theta_n, loss_function, batch_size, nb_layers, grad_threshold, layer_to_update, supervised, classifier_on_all_act) 
    





    """
    Calculates accuracy over a whole dataset with a cluster classifier
    """
    def cluster_accuracy(layers, x, y):

        @functools.partial(jax.jit, static_argnames=["nb_classes"])
        def predictions(layers_params, x, nb_classes):

            clusters_activity = None
            nb_neurons_clusters = int(layers_params[-1]["W"].shape[0] / nb_classes)
            
            for params_idx, params in enumerate(layers_params):
                x = jax.jit(ForwardModel.forward_one_pass)(x, params)
            

            for i in range (nb_classes):
                cluster_prediction = x[:, i * nb_neurons_clusters : (i+1) * nb_neurons_clusters].sum(axis = 1)
                clusters_activity = cluster_prediction if clusters_activity is None else jnp.vstack([clusters_activity, cluster_prediction])

            return jnp.argmax(clusters_activity.T, axis = -1), clusters_activity.T


        nb_classes = jnp.max(y).astype(jnp.int32).item() + 1
        layers_params = list(map(lambda layer : jam.eval_params(layer.Params), layers))

        y_predicted, guesses = predictions(layers_params, x, nb_classes)
        
        return jnp.mean(y == y_predicted), guesses
    
    """
    Calculates accuracy over a whole dataset with a softmax classifier
    """
    def softmax_accuracy(layers, x, y):

        @functools.partial(jax.jit)
        def predictions(layers_params, x):
            
            last_layer_feat = x

            for params_idx, params in enumerate(layers_params):
                if params_idx == len(layers_params) - 1:
                    x = jax.jit(ForwardModel.forward_with_softmax_one_pass)(last_layer_feat, params)
                else:
                    x = jax.jit(ForwardModel.forward_one_pass)(x, params)

                last_layer_feat = x

            return jnp.argmax(x, axis = -1), x

        layers_params = list(map(lambda layer : jam.eval_params(layer.Params), layers))

        y_predicted, guesses = predictions(layers_params, x)
        
        return jnp.mean(y == y_predicted), guesses
    

    """
    Calculates accuracy over a whole dataset with a softmax classifier (using all activations)
    """
    def softmax_accuracy_on_all_activations(layers, x, y):

        @functools.partial(jax.jit)
        def predictions(layers_params, x):
            
            last_layer_feat = None

            for params_idx, params in enumerate(layers_params):
                if params_idx == len(layers_params) - 1:
                    last_layer_feat = x if last_layer_feat is None else last_layer_feat
                    x = jax.jit(ForwardModel.forward_with_softmax_one_pass)(last_layer_feat, params)
                else:
                    x = jax.jit(ForwardModel.forward_one_pass)(x, params)

                last_layer_feat = x if last_layer_feat is None else jnp.concat([last_layer_feat, x], axis = -1)

            return jnp.argmax(x, axis = -1), x

        layers_params = list(map(lambda layer : jam.eval_params(layer.Params), layers))

        y_predicted, guesses = predictions(layers_params, x)
        
        return jnp.mean(y == y_predicted), guesses
    


    """
    Calculates accuracy over a whole dataset based on the higest goodness among all layers, tested for every label
    """
    def accuracy_goodness_swipe(layers, x, y, label_strength = 1):

        @jax.jit
        def compute_goodness(x: jnp.ndarray, params: dict) -> jnp.ndarray:
            """Compute goodness score as sum of squared activations."""
            activations = jax.jit(ForwardModel.forward_one_pass)(x, params)
            return jnp.power(activations, 2).sum(axis=1), activations
        

        nb_classes = jnp.max(y).astype(jnp.int32).item() + 1
        goodnesses = jnp.zeros((x.shape[0],nb_classes))
        for query_label in range(nb_classes):
            img = jnp.array(x.copy())
            img = img.at[:,:nb_classes].set(0)
            img = img.at[:,query_label].set(label_strength)

            for layer in layers:
                g, img = compute_goodness(img, jam.eval_params(layer.Params))
                goodnesses = goodnesses.at[:,query_label].add(g)

            y_predicted = jnp.argmax(goodnesses, axis = 1)

        return jnp.mean(y_predicted == y), goodnesses






    
    def train(
        layers : List[LayerParams],
        batch_size: int,
        train_imgs,
        train_labels,
        test_imgs,
        test_labels,
        supervised = False,
        classifier_on_all_act = True, # default softmax header spanning over all the hidden activations
        plot_reciptive_fields = False,
        acc_each_epoch = True
        ):

        nb_classes = jnp.max(train_labels).astype(jnp.int32).item() + 1

        # neutralized the test images and train images if supervised
        if supervised:
            train_imgs = ForwardModel.generate_neutral_batch(train_imgs, jnp.eye(nb_classes)[train_labels.astype(jnp.int32)])
            test_imgs = ForwardModel.generate_neutral_batch(test_imgs, jnp.eye(nb_classes)[test_labels.astype(jnp.int32)])

        x_batches = jax.device_put(batch(train_imgs, batch_size))
        targets_onehot = jax.device_put(jnp.eye(jnp.max(train_labels.astype(jnp.int32)).item() + 1)[train_labels.astype(jnp.int32)])
        targets_onehot_batches = jax.device_put(batch(targets_onehot, batch_size))


        loss_hists = []

        print(f"Start training ")

        test_acc_all = []
        train_acc_all = []
        pos_pulses_all = []
        neg_pulses_all = []


        # ravel params
        params = {}
        for layer_idx, layer in enumerate(layers):
            params[f"W{layer_idx}"] = layer.Params["W"]
            params[f"b{layer_idx}"] = layer.Params["b"]

        # check that we did not reach the 5000 pulses threshold
        overflow_memristor = 0

        for current_layer_idx, current_layer in enumerate(layers):

            learning_rate = current_layer.LearningRate
            theta_p = current_layer.PosThetaThreshold
            theta_n = current_layer.NegThetaThreshold
            threshold_grad = current_layer.ThresholdGradient
            loss_function = current_layer.LossFunction

            pbar = tqdm(range(current_layer.Epochs), desc=f"{current_layer_idx}")
            for epoch in pbar:

                loss_sum = 0.            

                for x_batch, label_batch in zip(x_batches, targets_onehot_batches):

                    loss, params = ForwardModel.step(x_batch, label_batch, params, theta_p, theta_n, loss_function, batch_size, len(layers), learning_rate, threshold_grad, current_layer_idx, supervised, classifier_on_all_act)
                    loss_sum += loss


                    # check for overflow on memristor carac after each batch
                    if current_layer.MemristorWeights :
                        if jnp.max(params[f"W{current_layer_idx}"].pos_pulse_id) > 5000 :
                            print("Pos pulses reached 5000")
                            params[f"W{current_layer_idx}"] = params[f"W{current_layer_idx}"].reset()
                            overflow_memristor = 1
                            break
                        if jnp.max(params[f"W{current_layer_idx}"].neg_pulse_id) > 5000 :
                            print("Neg pulses reached 5000")
                            params[f"W{current_layer_idx}"] = params[f"W{current_layer_idx}"].reset()
                            overflow_memristor = 1
                            break

                current_layer.update_params({
                    "W" : params[f"W{current_layer_idx}"], 
                    "b" : params[f"b{current_layer_idx}"]
                })

                # CLUSTER CLASSIFIER
                if layers[-1].LossFunction != "SoftMax":
                    # accuracy on clusters over layers
                    if (epoch == current_layer.Epochs - 1 and current_layer_idx == len(layers) - 1) or acc_each_epoch == True :
                        train_acc = ForwardModel.cluster_accuracy(layers, train_imgs, train_labels)[0]
                        test_acc = ForwardModel.cluster_accuracy(layers, test_imgs, test_labels)[0]
                    # compute only the final accuracy (faster simulations)
                    else :
                        train_acc = jnp.array(0)
                        test_acc = jnp.array(0)
                
                # SOFTMAX CLASSIFIER 
                else:

                    if classifier_on_all_act :
                        train_acc = ForwardModel.softmax_accuracy_on_all_activations(layers, train_imgs, train_labels)[0]
                        test_acc = ForwardModel.softmax_accuracy_on_all_activations(layers, test_imgs, test_labels)[0]
                    # compute only the final accuracy (faster simulations)
                    else:
                        train_acc = ForwardModel.softmax_accuracy(layers, train_imgs, train_labels)[0]
                        test_acc = ForwardModel.softmax_accuracy(layers, test_imgs, test_labels)[0]


                # check for memristor characteristic overflow after each epoch -> set the accuracy to 0 if the limit is reached 
                if current_layer.MemristorWeights :

                    if overflow_memristor:
                        train_acc = jnp.array(0.)
                        test_acc = jnp.array(0.)
                        warnings.warn(
                            "Memristor characteristic overflow detected: "
                            "train/test accuracy set to 0.",
                            category=RuntimeWarning
                        )
                    
                    pos_pulses = jnp.array(current_layer.Params['W'].pos_pulse_id).ravel()
                    neg_pulses = jnp.array(current_layer.Params['W'].neg_pulse_id).ravel()
                    pos_pulses_all.append(pos_pulses)
                    neg_pulses_all.append(neg_pulses)
                else :
                    pos_pulses_all.append(jnp.zeros(layer.Params['W'].shape).ravel())
                    neg_pulses_all.append(jnp.zeros(layer.Params['W'].shape).ravel())

                train_acc_all.append(train_acc)
                test_acc_all.append(test_acc)

                loss_hists.append(loss_sum)
                pbar.set_postfix({"loss": f"{loss_sum:.2f}", "train": f"{train_acc*100:.2f}%", "test": f"{test_acc*100:.2f}%"})

            print(f"Layer {current_layer_idx + 1} Trained : Test Accuracy : {test_acc*100}%")
            print(f"Layer {current_layer_idx + 1} Trained : Train Accuracy : {train_acc*100}%\n")

        return layers, test_acc_all, train_acc_all, loss_hists, pos_pulses_all, neg_pulses_all
        

























"""
Layer-wise Backprop algorithm class
"""
class BackPropagatedModel:

    @staticmethod
    def normalize(x: jnp.array):
        # no normalization by the L2 norm as done in original paper
        return x # / (jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True)) + 1e-6)
    

    @functools.partial(jax.vmap, in_axes=(0, None, None))
    @functools.partial(jax.jit, static_argnames=['nb_layers'])
    def forward(x, weights, nb_layers):
        weights = jam.eval_params(weights)

        for idx in range(nb_layers-1):
            x = BackPropagatedModel.normalize(x)
            x = jax.nn.relu(weights[f"W{idx}"] @ x)# + weights[f"b{idx}"])
        
        x = BackPropagatedModel.normalize(x)
        x = jax.nn.softmax(weights[f"W{nb_layers-1}"] @ x)# + weights[f"b{nb_layers-1}"])
        
        return x



    @functools.partial(jax.jit)
    def loss_fn(y, t):
        # return jnp.mean((y - t) ** 2) # MSE
        return -jnp.mean(jnp.sum(t * jnp.log(y + 1e-10), axis=-1)) # cross entropy loss
    
    @functools.partial(jax.value_and_grad, argnums=1)  # Take gradient wrt argument num 1
    def compute_grad(x, weights, targets_onehot, nb_layers):
        return BackPropagatedModel.loss_fn(BackPropagatedModel.forward(x, weights, nb_layers), targets_onehot)
    
    @functools.partial(jax.jit, static_argnames=['batch_size', 'nb_layers', 'learning_rate', 'grad_threshold', 'layer_to_update'])
    def step(imgs, targets_onehot, params, batch_size, nb_layers, learning_rate, grad_threshold, layer_to_update: int = -1):

        @functools.partial(jax.jit, static_argnames=['nb_layers', 'batch_size', 'grad_threshold', 'layer_to_update'])
        def update(imgs, targets_onehot, params, batch_size, nb_layers, grad_threshold, layer_to_update : int = -1):

            # used specifically for the last batch, in case the train size is not multiple of the batch size
            batch_size_factor =  imgs.shape[0] / batch_size
            loss, grads = BackPropagatedModel.compute_grad(imgs, jam.eval_params(params), targets_onehot, nb_layers)


            for grad_idx in range(nb_layers): 
                grads[f"W{grad_idx}"] = jnp.zeros(grads[f"W{grad_idx}"].shape) if grad_idx != layer_to_update and layer_to_update != -1 else grads[f"W{grad_idx}"]
                grads[f"b{grad_idx}"] = jnp.zeros(grads[f"b{grad_idx}"].shape) if grad_idx != layer_to_update and layer_to_update != -1 else grads[f"b{grad_idx}"]


            def weight_update(weight, grad):
                if isinstance(weight, jam.MemristorWeight):
                    return weight.incr_all(grad < -grad_threshold).decr_all(grad > grad_threshold)
                else:
                    return jax.tree_util.tree_map(lambda a, b: a - learning_rate * batch_size_factor * b, weight, grad)

            return loss, jam.map_params(weight_update, params, grads)
        
        return update(imgs, targets_onehot, params, batch_size, nb_layers, grad_threshold, layer_to_update) 
    


    @functools.partial(jax.jit, static_argnames=['nb_layers'])
    def accuracy(layers_params, data, labels, nb_layers):
        corrects = 0

        y = BackPropagatedModel.forward(data, layers_params, nb_layers)

        corrects += jnp.count_nonzero(
            jnp.argmax(y, axis=-1) == labels
        )

        return corrects / labels.shape[0]

    
    def train(
        layers : List[LayerParams],
        batch_size: int,
        train_imgs,
        train_labels,
        test_imgs,
        test_labels):

        x_batches = jax.device_put(batch(train_imgs, batch_size))
        targets_onehot = jax.device_put(jnp.eye(jnp.max(train_labels.astype(jnp.int32)).item() + 1)[train_labels.astype(jnp.int32)])
        targets_onehot_batches = jax.device_put(batch(targets_onehot, batch_size))

        loss_hists = []

        print(f"Start training ")

        test_acc_all = []
        train_acc_all = []
        pos_pulses_all = []
        neg_pulses_all = []

        overflow_memristor = 0


        # inverted trainin order, from last to first
        for current_layer_idx, current_layer in zip(range(len(layers) - 1, -1, -1), layers[::-1]):

            learning_rate = current_layer.LearningRate
            threshold_grad = current_layer.ThresholdGradient

            pbar = tqdm(range(current_layer.Epochs), desc=f"Training Layer {current_layer_idx + 1} with BackProp")
            for epoch in pbar:

                loss_sum = 0.
                params = {}

                for layer_idx, layer in enumerate(layers):
                    params[f"W{layer_idx}"] = layer.Params["W"]
                    params[f"b{layer_idx}"] = layer.Params["b"]
            

                for x_batch, label_batch in zip(x_batches, targets_onehot_batches):
                    loss, params = BackPropagatedModel.step(x_batch, label_batch, params, batch_size, len(layers), learning_rate, threshold_grad, current_layer_idx)
                    loss_sum += loss


                    # check for overflow on memristor carac after each batch
                    if current_layer.MemristorWeights :
                        if jnp.max(params[f"W{current_layer_idx}"].pos_pulse_id) > 5000 :
                            print("Pos pulses reached 5000")
                            params[f"W{current_layer_idx}"] = params[f"W{current_layer_idx}"].reset()
                            overflow_memristor = 1
                            break
                        if jnp.max(params[f"W{current_layer_idx}"].neg_pulse_id) > 5000 :
                            print("Neg pulses reached 5000")
                            params[f"W{current_layer_idx}"] = params[f"W{current_layer_idx}"].reset()
                            overflow_memristor = 1
                            break


                for layer_idx, layer in enumerate(layers):
                    layer.update_params({
                        "W" : params[f"W{layer_idx}"], 
                        "b" : params[f"b{layer_idx}"]
                    })

                loss_hists.append(loss_sum)

                # check for memristor characteristic overflow after each epoch -> set the accuracy to 0 if the limit is reached 
                if current_layer.MemristorWeights :

                    if overflow_memristor:
                        train_acc = jnp.array(0.)
                        test_acc = jnp.array(0.)
                        warnings.warn(
                            "Memristor characteristic overflow detected: "
                            "train/test accuracy set to 0.",
                            category=RuntimeWarning
                        )
                    
                    pos_pulses = jnp.array(current_layer.Params['W'].pos_pulse_id).ravel()
                    neg_pulses = jnp.array(current_layer.Params['W'].neg_pulse_id).ravel()
                    pos_pulses_all.append(pos_pulses)
                    neg_pulses_all.append(neg_pulses)
                else :
                    pos_pulses_all.append(jnp.zeros(layer.Params['W'].shape).ravel())
                    neg_pulses_all.append(jnp.zeros(layer.Params['W'].shape).ravel())
                
                params = {}
                for idx, layer in enumerate(layers):
                    params[f"W{idx}"] = layer.Params["W"]
                    params[f"b{idx}"] = layer.Params["b"]


                train_acc = BackPropagatedModel.accuracy(params, train_imgs, train_labels, len(layers))
                train_acc_all.append(train_acc)
                test_acc = BackPropagatedModel.accuracy(params, test_imgs, test_labels, len(layers))
                test_acc_all.append(test_acc)

                pbar.set_postfix({"loss": f"{loss_sum:.2f}", "train": f"{train_acc*100:.2f}%", "test": f"{test_acc*100:.2f}%"})

                if current_layer.MemristorWeights :
                    pos_pulses = jnp.array(current_layer.Params['W'].pos_pulse_id).ravel()
                    neg_pulses = jnp.array(current_layer.Params['W'].neg_pulse_id).ravel()
                    pos_pulses_all.append(pos_pulses)
                    neg_pulses_all.append(neg_pulses)
                else :
                    pos_pulses_all.append(jnp.zeros(layer.Params['W'].shape).ravel())
                    neg_pulses_all.append(jnp.zeros(layer.Params['W'].shape).ravel())

            print(f"Layer {current_layer_idx + 1} Trained : Test Accuracy : {test_acc*100}%")
            print(f"Layer {current_layer_idx + 1} Trained : Train Accuracy : {train_acc*100}%\n")

        return layers, test_acc_all, train_acc_all, loss_hists, pos_pulses_all, neg_pulses_all