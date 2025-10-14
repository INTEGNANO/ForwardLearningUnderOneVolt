from torch.utils.data import DataLoader
from torchvision import transforms as v2
import torch
import numpy as np

from lvr.drivers import b1530, psu
from lvr.nn import dataset

import json
from datetime import datetime
from pathlib import Path
from typing import Generator
from time import sleep

### GLOBALS ###
try:
    psu1 = psu.Keith2230G(
        adress="GPIB1::2::INSTR", reset=0, silence_initial_measurements=1
    )
    psu2 = psu.Keith2230G(
        adress="GPIB1::5::INSTR", reset=0, silence_initial_measurements=1
    )
except psu.visa.VisaIOError:
    print("[WARNING] Power Supplies not connected")
    psu1 = psu2 = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(num_features, *, ternarize=False):
    dset_config = dataset.AnimalsConfig(num_features=num_features)

    if ternarize:
        if num_features != 32:
            raise NotImplementedError(
                "Ternarization statistics is only computed for 32 features"
            )

        # fmt: off

        # Statistics performed beforehand on the training set and rounded to 2 significant digits
        # lower_bounds  = np.quantile(train_dataset.data, 0.33, dim=0)
        # higher_bounds = np.quantile(train_dataset.data, 0.66, dim=0)

        lower_bounds = np.array([0.29, 0.25, 0.22, 0.19, 0.53, 0.21, 0.37, 0.51, 0.22, 0.48, 0.14, 0.42, 0.97, 0.48, 0.48, 0.33, 0.31, 0.61, 0.42, 0.18, 0.24, 1.2, 0.28, 0.27, 0.24, 0.26, 0.22, 0.54, 0.07, 0.48, 0.28, 0.18])
        upper_bounds = np.array([0.62, 0.52, 0.50, 0.45, 0.95, 0.57, 0.68, 0.99, 0.46, 0.83, 0.35, 0.86, 1.4, 0.75, 0.86, 0.66, 0.59, 0.99, 0.76, 0.46, 0.51, 1.4, 0.56, 0.59, 0.49, 0.62, 0.50, 0.88, 0.56, 0.87, 0.54, 1.1])
        # fmt: on

        # Using numpy because AnimalsDataset uses numpy arrays
        transform = v2.Compose(
            [
                np.log1p,
                v2.Lambda(lambda x: np.where(x < lower_bounds, -1.0, x)),
                v2.Lambda(lambda x: np.where(x > upper_bounds, 1.0, x)),
                v2.Lambda(
                    lambda x: np.where(
                        (x >= lower_bounds) & (x <= upper_bounds), 0.0, x
                    )
                ),
            ]
        )

    else:
        transform = v2.Compose([])

    # Actually, test/val have been confused in the first place
    # In pratice, test is val and val is test
    # Name corrected in code, name not corrected in dataset file for consistency with older versions of the code
    train_dataset = dataset.AnimalsDataset(
        dset_config, split="train", transform=transform
    )
    val_dataset = dataset.AnimalsDataset(dset_config, split="test", transform=transform)
    test_dataset = dataset.AnimalsDataset(
        dset_config, split="validation", transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    return train_loader, val_loader, test_loader


def setup_logging(hyperparams: dict | str | Path, from_epoch=0):
    def _setup_layerwise_params(param):
        if not isinstance(hyperparams[param], list):
            hyperparams[param] = [hyperparams[param]] * (len(hyperparams["arch"]) - 1)

        elif len(hyperparams["arch"]) - 1 != len(hyperparams[param]):
            raise ValueError(
                f"Number of layers ({len(hyperparams['arch']) - 1}) and len of '{param}' {len(hyperparams[param])} not matching"
            )

    if isinstance(hyperparams, (str, Path)):
        # Doing inference for a previous run
        LOGDIR = Path(hyperparams)
        with open(LOGDIR / "hyperparams.json", "r") as f:
            hyperparams = json.load(f)

        if from_epoch == 0:
            hyperparams["num_epochs"] = 0  # Set inference mode

        # Because params has been added later, may no be present
        hyperparams["array_offset"] = hyperparams.get("array_offset", (0, 0))
        hyperparams["learning_rule"] = hyperparams.get("learning_rule", "bp")
        hyperparams["hardware_mac"] = hyperparams.get("hardware_mac", False)
        hyperparams["chip"] = hyperparams.get("chip", "aa")

        return hyperparams, LOGDIR

    hyperparams["array_size"] = hyperparams.get("array_size", (64, 64))
    hyperparams["array_offset"] = hyperparams.get("array_offset", (0, 0))
    hyperparams["learning_rule"] = hyperparams.get("learning_rule", "bp")
    hyperparams["hardware_mac"] = hyperparams.get("hardware_mac", False)
    hyperparams["chip"] = hyperparams.get("chip", "aa")

    ## Check
    for hp in [
        "arch",
        # "array_size",
        # "array_offset",
        # "learning_rule",
        # "hardware_mac",
        # "memory_map",
        "scale_factor",
        "grad_thresh",
        "num_epochs",
        "switch_epoch",
    ]:
        if hp not in hyperparams:
            raise ValueError(f"Expected '{hp}' in hyperparams")

    if (
        not hyperparams.get("fp32_training", False)
        and "simexp" not in hyperparams
        and "memory_map" not in hyperparams
    ):
        raise ValueError("Doing experiments, but no `memory_map` specified")

    if hyperparams.get("fp32_training", False) and "learning_rate" not in hyperparams:
        raise ValueError("Training using FP32, but no 'learning_rate' specified")

    if hyperparams["hardware_mac"] and hyperparams["chip"] != "sk":
        raise ValueError(
            "Hardware MAC is only implemented for SwissKnife chip ('sk'), not for AwesomeArray ('aa')"
        )

    if (
        ("simexp" in hyperparams)
        + hyperparams["hardware_mac"]
        + hyperparams["fp32_training"]
    ) > 1:
        raise ValueError(
            "Only one of the three options can be set to True: `hardware_mac`, `fp32_training`, or `simexp`"
        )

    if hyperparams["hardware_mac"] and not hyperparams.get("ternarize", False):
        raise ValueError(
            "Hardware MAC is only implemented for ternarized inputs, please set `ternarize` to True"
        )

    _setup_layerwise_params("scale_factor")
    _setup_layerwise_params("grad_thresh")

    if hyperparams["learning_rule"] != "bp":  # forward forward rule
        _setup_layerwise_params("ff_thetas")

    ## Create
    timestamp = datetime.now().strftime("%d%m_%H%M%S")
    LOGDIR = Path(f"data/run_logs/{timestamp}")
    LOGDIR.mkdir(parents=True)
    (LOGDIR / "array_conductances").mkdir(parents=True)
    (LOGDIR / "model_params").mkdir(parents=True)
    (LOGDIR / "grads").mkdir(parents=True)
    (LOGDIR / "raw_grads").mkdir(parents=True)
    with open(LOGDIR / "updates.csv", "w") as f:
        f.write("epoch,step,update_count\n")

    with open(LOGDIR / "hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=2)

    with open(LOGDIR / "metrics.csv", "w") as f:
        # Actually, test/val have been confused in the first place
        # In pratice, test is val and val is test
        # Corrected in code, not corrected in log for consistency with already-made runs
        f.write("epoch,train_loss,train_acc,test_acc,val_acc\n")

    return hyperparams, LOGDIR


def load_memory_map(filepath, model, device):
    with open(filepath, "r") as f:
        arrays = f.read().strip().split("=====")

    source_idxs = [
        torch.Tensor(
            [
                tuple(map(int, el.split(",")))
                for el in c.strip().replace("\n", " ").split(" ")
            ],
            device=device,
        ).to(int)
        for c in arrays
    ]

    target_idxs = [
        torch.Tensor(
            [(i, j) for i in range(p.size(0)) for j in range(p.size(1))], device=device
        ).to(int)
        for p in model.parameters()
    ]

    return (target_idxs, source_idxs)


def create_mlp(arch, device, *, relu_at_end=False):
    layers = []
    for inp, out in zip(arch[:-2], arch[1:-1]):
        layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(inp, out, bias=False),
                torch.nn.ReLU(),
            )
        )

    if relu_at_end:
        layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(arch[-2], arch[-1], bias=False),
                torch.nn.ReLU(),
            )
        )
    else:
        layers.append(
            torch.nn.Sequential(torch.nn.Linear(arch[-2], arch[-1], bias=False))
        )

    model = torch.nn.Sequential(*layers)
    return model.to(device)


### Power Supplies Utils
def set_psu(chip):
    match chip:
        case "aa":
            psu1.set_channel_voltage("CH1", 5.0)  # VDD5
            psu1.set_channel_current("CH1", 0.1)
            psu1.set_channel_output("CH1", 1)
            psu2.set_channel_voltage("CH1", 1.2)  # VDDR
            psu2.set_channel_current("CH1", 0.01)
            psu2.set_channel_output("CH1", 1)
            psu2.set_channel_voltage("CH3", 1.2)  # VDDC
            psu2.set_channel_current("CH3", 0.01)
            psu2.set_channel_output("CH3", 1)
            psu2.set_channel_voltage("CH2", 1.2)  # VDD
            psu2.set_channel_current("CH2", 0.1)
            psu2.set_channel_output("CH2", 1)

        case "sk":
            psu1.set_channel_output("CH1", 0)
            psu2.set_channel_voltage("CH1", 5.0)  # VDD
            psu2.set_channel_current("CH1", 0.1)
            psu2.set_channel_output("CH1", 1)
            psu2.set_channel_voltage("CH2", 0.897)  # BL1
            psu2.set_channel_current("CH2", 0.01)
            psu2.set_channel_output("CH2", 1)
            psu2.set_channel_voltage("CH3", 0.7)  # BL2=V0
            psu2.set_channel_current("CH3", 0.01)
            psu2.set_channel_output("CH3", 1)

    sleep(2)


def turn_off_psu():
    psu1.set_channel_output("CH1", 0)
    psu2.set_channel_output("CH1", 0)
    psu2.set_channel_output("CH2", 0)
    psu2.set_channel_output("CH3", 0)


def to_torch(x, dtype=torch.float32):
    return torch.tensor(x, device=device).to(dtype)


def array_exec_and_fetch(
    array_size: tuple[int, int], *, current_name="I", voltage_name="Vcol"
):
    memristor_conductances = b1530.exec_and_fetch(
        current_name=current_name, voltage_name=voltage_name
    )[:-1].reshape(array_size[0], 2 * array_size[1])

    return memristor_conductances


def params_from_conductances(gs):
    return to_torch(gs[:, ::2]) - to_torch(gs[:, 1::2])


def map_memory(
    map: tuple[list[torch.Tensor]],
    array_offset: tuple[int, int],
    memory: torch.Tensor,
    parameters: Generator[torch.Tensor, None, None],
    *,
    inverse_map: bool = False,
    scale: float | list[float] = 1.0,
):
    """Map memory from memristor array to model parameters with scaling factor, or the opposite if `inverse_map` is `True`."""
    with torch.no_grad():
        for i, params in enumerate(parameters):
            target_idxs = map[0][i]
            source_idxs = map[1][i]
            scaling = scale[i] if isinstance(scale, list) else scale

            if inverse_map:
                memory[
                    source_idxs[:, 0] - array_offset[0],
                    source_idxs[:, 1] - array_offset[1],
                ] = scaling * params[target_idxs[:, 0], target_idxs[:, 1]]
            else:
                params[target_idxs[:, 0], target_idxs[:, 1]] = (
                    scaling
                    * memory[
                        source_idxs[:, 0] - array_offset[0],
                        source_idxs[:, 1] - array_offset[1],
                    ]
                )
