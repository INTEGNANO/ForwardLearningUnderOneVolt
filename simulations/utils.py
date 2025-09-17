
import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt # to remove todo

from scipy.signal import convolve2d

import random

import os
import json
import pandas as pd
from datetime import datetime

from scipy.stats import gaussian_kde

from sklearn.manifold import TSNE




def gen_dataset(train_x, train_y, test_x, test_y, supervised, rdm_key, label_strength = 1.0):

    def random_different_labels(true_labels, nb_class, rdm_key):
        random_labels = jax.random.randint(rdm_key, true_labels.shape, 0, nb_class)
        diff_labels = jnp.where(random_labels == true_labels, (random_labels + 1) % nb_class, random_labels)
        onehot_diff_labels = jnp.eye(nb_class)[diff_labels.astype(jnp.int32)]
        return onehot_diff_labels

    train_x     = jax.device_put(train_x)
    train_y     = jax.device_put(train_y)

    test_x      = jax.device_put(test_x)

    nb_class    = jnp.max(train_y).astype(jnp.int32).item() + 1

    if supervised:
        onehot_labels = jnp.eye(nb_class)[train_y.astype(jnp.int32)]
        
        test_x =  test_x.T  if jnp.all(test_x.T[:,:nb_class]  == 0) else jnp.hstack((jnp.zeros((test_x.shape[-1], nb_class)),  test_x.T))
        train_x = train_x.T if jnp.all(train_x.T[:,:nb_class] == 0) else jnp.hstack((jnp.zeros((train_x.shape[-1], nb_class)), train_x.T))

        test_x = test_x.at[:, :nb_class].set(label_strength/nb_class)
        train_pos_x = train_x.at[:, :nb_class].set(label_strength/nb_class)
        train_neg_x = train_x.at[:, :nb_class].set(label_strength/nb_class)
        train_x = train_x.at[:, :nb_class].set(label_strength/nb_class)


    else:
        test_x = test_x.T
        train_x = train_x.T
        train_pos_x = train_x
        train_neg_x = train_x


    return train_x, train_pos_x, train_neg_x, train_y, test_x, test_y



############################################################
##                      LOGS_MNGT                         ##

# Function to safely convert objects to JSON-friendly formats
def serialize(obj):
    if isinstance(obj, jnp.ndarray) or isinstance(obj, np.ndarray):
        return {"shape": obj.shape}  # Store only the shape of JAX/NumPy arrays
    elif isinstance(obj, (int, float, str, bool)):  
        return obj  # Already serializable
    elif isinstance(obj, tuple):  
        return [serialize(i) for i in obj]  # Convert tuples recursively into lists
    elif isinstance(obj, dict):  
        return {serialize(k): serialize(v) for k, v in obj.items()}  # Convert dict keys/values
    elif hasattr(obj, '__dict__'):  
        return {serialize(k): serialize(v) for k, v in vars(obj).items()}  # Convert objects
    return str(obj)  # Convert any other type to string



def plot_half_violin(ax, data, positions, side='right', width=2.0, color='#1f77b4', alpha=1.0):
    """Helper function to plot asymmetric half violins."""
    for d, pos in zip(data, positions):
        d = np.asarray(d)
        if len(d) == 0:
            continue
        if np.all(d == 0):  # Check if the array contains only zeros
            # Skip KDE and set the density to 0 (or any other desired behavior)
            continue
        else:
            kde = gaussian_kde(d)
            x = np.linspace(np.min(d), np.max(d), 100)
            y = kde(x)
            y = y / y.max() * width  # normalize width


        if side == 'right':
            ax.fill_betweenx(x, pos, pos + y, facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.5)
        elif side == 'left':
            ax.fill_betweenx(x, pos - y, pos, facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.5)


def save_logs(parameters, test_accuracy, train_accuracy, loss, positive_pulses, negative_pulses, title: str, plot=True, save_pulses=True):
    """
    Plots accuracy and pulse distributions with a modern, clear style.
    Saves .svg and .png versions of the plots.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    import os
    import json
    import pandas as pd
    from datetime import datetime
    from scipy.stats import gaussian_kde
    import jax.numpy as jnp

    plt.style.use('seaborn-v0_8-muted')
    matplotlib.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    # Convert to JAX arrays and transpose
    test_accuracy = jnp.array(test_accuracy).T
    train_accuracy = jnp.array(train_accuracy).T
    epochs = range(1, len(test_accuracy) + 1)
    loss = jnp.array(loss).T
       

    def compute_stats(acc):
        return [np.mean(acc[i]) for i in range(acc.shape[0])], \
               [np.min(acc[i]) for i in range(acc.shape[0])], \
               [np.max(acc[i]) for i in range(acc.shape[0])]

    mean_test, min_test, max_test = compute_stats(test_accuracy)
    mean_train, min_train, max_train = compute_stats(train_accuracy)
    mean_loss, _, _ = compute_stats(loss)

    mean_pulses = []

    for pos_col_group, neg_col_group in zip(zip(*positive_pulses), zip(*negative_pulses)):
        pos_mean = jnp.mean(jnp.concatenate([arr.flatten() for arr in pos_col_group]))
        neg_mean = jnp.mean(jnp.concatenate([arr.flatten() for arr in neg_col_group]))
        mean_pulses.append(jnp.array([(pos_mean + neg_mean)/2]))

    run_folder = f"runs/ANIMALS/run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(run_folder, exist_ok=True)

    pd.DataFrame({
        "Epoch": epochs,
        "min_test_acc": min_test, "max_test_acc": max_test, "mean_test_acc": mean_test,
        "min_train_acc": min_train, "max_train_acc": max_train, "mean_train_acc": mean_train,
        "mean_loss" : mean_loss,
        "avg_pulses_per_device" : mean_pulses
    }).to_csv(os.path.join(run_folder, "accuracy_data.csv"), index=False)
    
    with open(os.path.join(run_folder, "parameters.json"), "w") as f:
        json.dump(parameters, f, indent=4)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.1], height_ratios=[1, 1])

    if save_pulses:

        pos_pulses = [jnp.concatenate(group) for group in zip(*positive_pulses)]
        neg_pulses = [jnp.concatenate(group) for group in zip(*negative_pulses)]

        pulse_rows = []

        for epoch_idx, (pos_vec, neg_vec) in enumerate(zip(pos_pulses, neg_pulses), start=1):
            pulse_rows.append({
                "Type": "Positive",
                "Epoch": epoch_idx,
                "Pulse Value": pos_vec.tolist()
            })
            pulse_rows.append({
                "Type": "Negative",
                "Epoch": epoch_idx,
                "Pulse Value": neg_vec.tolist()
            })


        pd.DataFrame(pulse_rows).to_csv(os.path.join(run_folder, "pulse_distributions.csv"), index=False)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.errorbar(
        epochs, mean_test,
        yerr=[np.array(mean_test) - np.array(min_test),
              np.array(max_test) - np.array(mean_test)],
        fmt='-o', color='#C44E52', ecolor='#FF9999', capsize=3,
        linewidth=2, label='Test Accuracy'
    )
    ax1.errorbar(
        epochs, mean_train,
        yerr=[np.array(mean_train) - np.array(min_train),
              np.array(max_train) - np.array(mean_train)],
        fmt='-o', color='#4C72B0', ecolor='#A5C8E1', capsize=3,
        linewidth=2, label='Train Accuracy'
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy", color='black')
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlim(0, len(test_accuracy) + 2)
    ax1.set_title("Train/Test Accuracy")
    ax1.legend(loc='upper left')

    ax2 = fig.add_subplot(gs[1, 1])
    flat_test_acc = np.array(test_accuracy[-1, :]).flatten()
    flat_train_acc = np.array(train_accuracy[-1, :]).flatten()

    kde_test = gaussian_kde(flat_test_acc, bw_method='silverman')
    kde_train = gaussian_kde(flat_train_acc, bw_method='silverman')
    x_values = np.linspace(0.0, 1.0, 1000)

    ax2.plot(kde_test(x_values), x_values, label='Test Accuracy', color='#C44E52', linewidth=2)
    ax2.plot(kde_train(x_values), x_values, label='Train Accuracy', color='#4C72B0', linewidth=2)

    mean_test_acc = np.mean(flat_test_acc)
    std_test_acc = np.std(flat_test_acc)
    mean_train_acc = np.mean(flat_train_acc)
    std_train_acc = np.std(flat_train_acc)

    ax2.text(x=0.95 * max(kde_test(x_values)), y=0.3,
             s=f"μ={mean_test_acc:.4f}\nσ={std_test_acc:.4f}",
             color='#C44E52', va='center', ha='right', fontsize=11,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
    ax2.text(x=0.95 * max(kde_test(x_values)), y=0.1,
             s=f"μ={mean_train_acc:.4f}\nσ={std_train_acc:.4f}",
             color='#4C72B0', va='center', ha='right', fontsize=11,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

    ax2.set_xlabel('Density')
    ax2.set_yticks([])  # Remove y-ticks
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)


    ax3 = fig.add_subplot(gs[0, 1])

    # Combine all accuracy values
    combined_acc = np.concatenate([flat_test_acc])

    # Compute KDE and density-based quantiles
    kde = gaussian_kde(combined_acc, bw_method='silverman')
    x_vals = np.linspace(0.0, 1.0, 1000)
    pdf_vals = kde(x_vals)
    cdf_vals = np.cumsum(pdf_vals)
    cdf_vals /= cdf_vals[-1]  # Normalize

    # Find the accuracy thresholds at 20% and 80% quantiles
    q20_idx = np.searchsorted(cdf_vals, 0.2)
    q80_idx = np.searchsorted(cdf_vals, 0.8)
    acc_q20 = x_vals[q20_idx]
    acc_q80 = x_vals[q80_idx]

    # Histogram with density normalization
    bins = np.linspace(0.0, 1.0, 30)
    ax3.hist(flat_test_acc, bins=bins, alpha=0.6, label='Test Accuracy',
            color='#C44E52', edgecolor='white', density=True)
    # ax3.hist(flat_train_acc, bins=bins, alpha=0.6, label='Train Accuracy',
    #         color='#4C72B0', edgecolor='white', density=True)

    # Highlight quantile zones
    ax3.axvspan(0.0, acc_q20, color='gray', alpha=0.2, label='0–20% Density Quantile')
    ax3.axvspan(acc_q80, 1.0, color='gray', alpha=0.2, label='80–100% Density Quantile')

    # Add vertical lines and annotations
    ax3.axvline(acc_q20, color='black', linestyle='--')
    ax3.axvline(acc_q80, color='black', linestyle='--')
    ax3.text(acc_q20, ax3.get_ylim()[1] * 0.9, f'{acc_q20:.2f}', ha='right', va='top', fontsize=10, rotation=90)
    ax3.text(acc_q80, ax3.get_ylim()[1] * 0.9, f'{acc_q80:.2f}', ha='left', va='top', fontsize=10, rotation=90)

    ax3.set_yticks([])
    ax3.set_ylabel("")
    ax3.set_xlim(0.5, 1.0)
    ax3.set_xlabel("Test Accuracy")
    ax3.set_title("")
    # ax3.legend()
    ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
    

    fig.suptitle(title, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(wspace=0.02, hspace=0.4)

    plot_path_svg = os.path.join(run_folder, "accuracy_and_pulse_plot.svg")
    plot_path_png = os.path.join(run_folder, "accuracy_and_pulse_plot.png")
    plt.savefig(plot_path_svg)
    plt.savefig(plot_path_png, dpi=300)

    if plot:
        plt.show()
    else:
        plt.close(fig)