# Forward-only learning in memristor arrays with month-scale stability
Companion repository for `Forward-only learning in memristor arrays with month-scale stability` [unpublished yet; submitted].

We perform on-chip learning on 8,064 memristors using Forward-only learning rules.
This repository holds the experiments code used, as well as the more extensive simulations on MNIST and additional hyperparameters study.

## Environment installation
First, move to either `experiments` folder or `simulations` folder.
Code tested for Python >=3.10. It is preferable to use a Python venv, or Conda/Mamba/Micromamba. Once your virtual environment created, you can install the dependencies using:
```
pip install -r requirements.txt
```

### Dataset (re)generation
You need to install `transformers`. Go to `shared_data/animals` and run:
```
$ pip install -r additional_requirements.txt
$ python gen_features.py
preprocessor_config.json: 100%|███████████████████████████████| 266/266 [00:00<?, ?B/s]
# ...
```

## `experiments`
This folder contains the code used for experiments, as well as the "simexp" code. This "simexp" setup is using replays of measured low-voltage reset
trajectories of 5,000 pulses for 1,268 distinct memristors (which can be found in `shared_data/lowvolteresets_measurements/base.hdf5`).
If you are in the `experiments` folder, utilities functions are made to run the simexp for Supervised Forward-Forward or Competitive Forward with the same hyperparameters used as the experiments:
```bash
$ python -i lvr
[WARNING] Power Supplies not connected
[WARNING] Power Supplies not connected
>>> run_cf_exps(simexp=True)
100%|█████████████████████████████████████████████████████████████| 250/250 [00:03<00:00, 69.26it/s]
Computing Train Accuracy...
Computing Test Accuracy...
Epoch 1, Loss: 1.3203, Train Acc: 21.20%, Test Acc: 20.54%
# ...
===== FINISHED =====
>>> run_sff_exps(simexp=True)
# ...
===== FINISHED =====
>>> exit()
$
```

Then, the logged data can be found in `experiments/data/run_logs/DDMM_HHMMSS/`.

Implementations of the learning rules can be found in [`experiments/lvr/nn/learning_rules.py`](experiments/lvr/nn/learning_rules.py).

Furthermore, all the figures and additional data of the paper have been generated using the Notebooks `experiments/plot_*`, in particular [`experiments/plot_experiments.ipynb`](experiments/plot_experiments.ipynb)

## `simulations`
This folder contains the code used for simulation studies. You can run the code for either Backprop, Supervised Forward-Forward, or Competitive Forward (GPU required):

```bash
$ python animals_cf.py
```