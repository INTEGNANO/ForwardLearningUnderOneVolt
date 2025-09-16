import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from drivers import b1530, uc_aa, uc_sk
import nn.learning_rules as lr
from nn.hardware_linear import HardwareLinear
import simexp
import utils

from time import sleep
from datetime import datetime

### GLOBALS ###
device = utils.device
uc = None  # Will be uc_aa or uc_sk depending on the chip used


def config_sweep_mask(uc_driver, synapse_grads, array_size):
    def _synase_grad_to_bit_mask(g):
        """
        Converts a gradient to the mask selecting which memristor to apply a low voltage reset to.

        The formula is W = alpha * (conductance_mem - conductance_compl_mem). A low voltage reset reduces the conductance.
        * If grad > 0, we want to decrease W, thus decrease `conductance_mem` and the selected bit should be the LSB
        * If grad < 0, we want to increase W, thus decrease `conductance_compl_mem` and the selected bit should be the MSB
        * If |grad| == 0, we dont want to modify W
        """
        if g > 0:
            return 0b01
        elif g < 0:
            return 0b10
        else:
            return 0b00

    # The mask is specified column-wise here
    # There are 2*X memristor per column (X=64 for AwesomeArray, X=32 for SwissKnife)
    # The mask is sent to the µc by packets of 4 rows w/ 2 bits per row (the LSB for the mem. and the MSB for its compl.)
    # Therefore, N x (2x4) = 2xX bits words must be sent to the µc
    for j in range(array_size[1]):
        col_mask = [0b000000] * 16
        for i in range(array_size[0] // 4):
            col_synapse_grads = synapse_grads[4 * i : 4 * (i + 1), j]
            col_mask[i] |= _synase_grad_to_bit_mask(col_synapse_grads[0]) << 0
            col_mask[i] |= _synase_grad_to_bit_mask(col_synapse_grads[1]) << 2
            col_mask[i] |= _synase_grad_to_bit_mask(col_synapse_grads[2]) << 4
            col_mask[i] |= _synase_grad_to_bit_mask(col_synapse_grads[3]) << 6

        uc_driver.set_sweep_mask(uc.SetSweepMask.COL, j, *col_mask)


def config_read_write(array_size, b1530_init_delay, pattern_default):
    b1530.dll.clear()
    b1530.setup(
        [
            ("WEAK_RESET", pattern_default),
            ("READ", pattern_default),
        ],
        2 * array_size,
        en_trig=True,
        overwrite_init_delay=b1530_init_delay,
    )


def run_inference_from_checkpoint(hyperparams: str, chkpt: str):
    """Perform inference using weights stored from previous experiment. Fully Software-side procedure."""
    hyperparams, LOGDIR = utils.setup_logging(hyperparams)
    num_features = (
        hyperparams["arch"][0]
        if hyperparams["learning_rule"] != "sff"
        else hyperparams["arch"][0] - 4
    )
    train_loader, test_loader, val_loader = utils.load_dataset(
        num_features, ternarize=hyperparams.get("ternarize", False)
    )

    model = utils.create_mlp(
        hyperparams["arch"], device, relu_at_end=hyperparams["learning_rule"] != "bp"
    )

    match hyperparams["learning_rule"]:
        case "bp":
            compute_accuracy = lr.compute_accuracy_bp
        case "sff":
            compute_accuracy = lr.compute_accuracy_ff
        case "cf":
            compute_accuracy = lr.compute_accuracy_ff
        case _:
            raise ValueError(f"Unknown learning rule: {hyperparams['learning_rule']}")

    MEMORY_MAP = utils.load_memory_map(hyperparams["memory_map"], model, device)

    memristor_conductances = np.load(
        (LOGDIR / "array_conductances" / chkpt).with_suffix(".npy")
    )
    memristive_parameters = utils.params_from_conductances(memristor_conductances)

    utils.map_memory(
        MEMORY_MAP,
        hyperparams["array_offset"],
        memristive_parameters,
        model.parameters(),
        scale=hyperparams["scale_factor"],
    )

    train_acc = compute_accuracy(model, train_loader)
    test_acc = compute_accuracy(model, test_loader)
    val_acc = compute_accuracy(model, val_loader)

    print(
        f"Train Acc: {train_acc * 100:.2f}%, Test Acc: {test_acc * 100:.2f}%, Val Acc: {val_acc * 100:.2f}%"
    )


def run_training(hyperparams, *, from_epoch=0):
    global uc

    hyperparams, LOGDIR = utils.setup_logging(hyperparams, from_epoch)

    ### SOFTWARE ###
    # ff encodes oh labels in first 4 inputs thus 4 less features are used
    num_features = (
        hyperparams["arch"][0]
        if hyperparams["learning_rule"] != "sff"
        else hyperparams["arch"][0] - 4
    )
    train_loader, test_loader, val_loader = utils.load_dataset(
        num_features, ternarize=hyperparams.get("ternarize", False)
    )

    ARCH = np.array(hyperparams["arch"])
    ARRAY_SIZE = hyperparams["array_size"][0] * hyperparams["array_size"][1]

    if simexp_filename := hyperparams.get("simexp"):
        simexp_pool = simexp.Pool(
            ARCH, simexp_filename, scale_factor=hyperparams["scale_factor"]
        )
    else:
        simexp_pool = None
    ### ###

    ### HARDWARE INIT ###
    if simexp_pool is None:
        utils.set_psu(hyperparams["chip"])
        b1530.init()

        match hyperparams["chip"]:
            case "aa":
                uc = uc_aa

                PATTERN_DEFAULT = "default"
                B1530_INIT_DELAY = 110e-6
                CURRENT_CHANNEL_NAME = "I"
                VOLTAGE_CHANNEL_NAME = "Vcol"
            case "sk":
                uc = uc_sk

                PATTERN_DEFAULT = "default_sk"
                B1530_INIT_DELAY = 175e-6
                CURRENT_CHANNEL_NAME = "VSL0"
                VOLTAGE_CHANNEL_NAME = "BL0"

        uc_driver = uc.MCDriver(
            hyperparams["array_size"], hyperparams["array_offset"], mask_to_ones=False
        )

        ack_mode = uc.ACK.FILL_SRs | uc.ACK.SET_SWEEP_MASK
        if hyperparams["chip"] == "sk":
            ack_mode |= uc.ACK.SET_CS_EN
        uc_driver.ack_mode(ack_mode)
    ### ###

    ### MODEL SETUP ###
    if hyperparams["hardware_mac"]:
        if hyperparams["arch"] != [32, 4]:
            raise NotImplementedError(
                "Hardware MAC is only implemented for 32-4 architecture"
            )

        if hyperparams["learning_rule"] != "bp":
            raise NotImplementedError(
                "Hardware MAC is only implemented for BP learning rule, not for "
                f"{hyperparams['learning_rule']}"
            )

        model = torch.nn.Sequential(
            HardwareLinear(
                in_features=ARCH[0],
                out_features=ARCH[1],
                uc_driver=uc_driver,
                scale_factor=hyperparams["scale_factor"][0],
                device=device,
            )
        )

        compute_accuracy = lr.compute_accuracy_bp
        compute_grads = lr.compute_grads_hardware

    else:
        model = utils.create_mlp(
            ARCH, device, relu_at_end=hyperparams["learning_rule"] != "bp"
        )

        match hyperparams["learning_rule"]:
            case "bp":
                compute_accuracy = lr.compute_accuracy_bp
                compute_grads = lr.compute_grads_bp
            case "sff":
                compute_accuracy = lr.compute_accuracy_ff
                compute_grads = lr.compute_grads_sff
            case "cf":
                compute_accuracy = lr.compute_accuracy_ff
                compute_grads = lr.compute_grads_cf
            case _:
                raise ValueError(
                    f"Unknown learning rule: {hyperparams['learning_rule']}"
                )
    ### ###

    if simexp_pool is not None:
        simexp_pool.update_model_params(model)
    else:
        ### HARDWARE <-> SOFTWARE MODEL INIT ###
        MEMORY_MAP = utils.load_memory_map(hyperparams["memory_map"], model, device)

        # Read initial weights from array
        b1530.setup(
            [("READ", PATTERN_DEFAULT)],
            2 * ARRAY_SIZE,
            en_trig=True,
            overwrite_init_delay=B1530_INIT_DELAY,
        )
        uc_driver.array_sweep(0)
        memristor_conductances = utils.array_exec_and_fetch(
            hyperparams["array_size"],
            current_name=CURRENT_CHANNEL_NAME,
            voltage_name=VOLTAGE_CHANNEL_NAME,
        )
        memristive_parameters = utils.params_from_conductances(memristor_conductances)

        # Software-side mirror of memristor array used to store gradients before applying updates
        synapse_grads = torch.zeros_like(memristive_parameters, device=device)

        # When hardware_mac, no need to map hardware weights to software
        # as no need to do software MAC
        # Still read and logged for debugging purposes
        utils.map_memory(
            MEMORY_MAP,
            hyperparams["array_offset"],
            memristive_parameters,
            model.parameters(),
            scale=hyperparams["scale_factor"],
        )

        # num_epochs == 0 <=> we just want to do inference with hardware weights
        if hyperparams["num_epochs"] == 0:
            np.save(
                LOGDIR
                / "array_conductances"
                / f"after_{datetime.now().strftime('%d%m_%H%M')}.npy",
                memristor_conductances,
            )
        elif from_epoch == 0:
            np.save(LOGDIR / "array_conductances" / "init.npy", memristor_conductances)

        # Setup B1530 for MAC (inference) is required, otherwise only write then read
        if hyperparams["hardware_mac"]:
            b1530.dll.clear()
            b1530.setup([("MAC", PATTERN_DEFAULT)], 1)
        else:
            config_read_write(ARRAY_SIZE, B1530_INIT_DELAY, PATTERN_DEFAULT)
        ### ###

    ### INFERENCE ###
    if hyperparams["num_epochs"] == 0:
        print("Computing Train Accuracy...")
        train_acc = compute_accuracy(model, train_loader)
        print("Computing Test Accuracy...")
        test_acc = compute_accuracy(model, test_loader)
        print("Computing Validation Accuracy...")
        val_acc = compute_accuracy(model, val_loader)
        with open(LOGDIR / "metrics.csv", "a") as f:
            f.write(
                f"{int(datetime.now().strftime('%d%m')):04},{int(datetime.now().strftime('%H%M')):04},{train_acc},{test_acc},{val_acc}\n"
            )
        print(
            f"Train Acc: {train_acc * 100:.2f}%, Test Acc: {test_acc * 100:.2f}%, Val Acc: {val_acc * 100:.2f}%"
        )

        b1530.dll.clear()
        b1530.dll.closeSession()
        uc_driver.ser.close()
        utils.turn_off_psu()
        return LOGDIR

    ### TRAINING ###
    if from_epoch > 0:
        print(f"Starting back training from epoch #{from_epoch + 1}...")
        input("Run? ")

    for epoch in range(from_epoch, hyperparams["num_epochs"]):
        running_loss = 0.0
        for step, (feats, labels) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            model.zero_grad(False)
            loss = compute_grads(model, feats, labels, hyperparams.get("ff_thetas"))

            for layer, p in enumerate(model.parameters()):
                np.save(LOGDIR / "raw_grads" / f"e{epoch}s{step}l{layer}.npy", p.grad)

            # Layerwise Learning
            if len(ARCH) >= 3:
                for i, p in enumerate(model.parameters()):
                    # Layer frozen = not learnt while the other is
                    layer_freezing_order = (
                        [0, 1] if hyperparams["learning_rule"] == "bp" else [1, 0]
                    )
                    if (
                        i == layer_freezing_order[0]
                        and epoch < hyperparams["switch_epoch"]
                    ):
                        p.grad.zero_()
                    elif (
                        i == layer_freezing_order[1]
                        and epoch >= hyperparams["switch_epoch"]
                    ):
                        p.grad.zero_()

            # Selective Weight Update: Zero-out the gradients whose mag. is below the threshold
            for param, thresh in zip(model.parameters(), hyperparams["grad_thresh"]):
                param.grad *= param.grad.abs() >= thresh

            if simexp_pool is not None:
                for layer, p in enumerate(model.parameters()):
                    np.save(LOGDIR / "grads" / f"e{epoch}s{step}l{layer}.npy", p.grad)
            else:
                # Map gradients back to software-side mirror of memristor array
                synapse_grads.zero_()
                utils.map_memory(
                    MEMORY_MAP,
                    hyperparams["array_offset"],
                    synapse_grads,
                    map(lambda x: x.grad, model.parameters()),
                    inverse_map=True,
                )

                np.save(LOGDIR / "grads" / f"e{epoch}s{step}.npy", synapse_grads)

            num_updates = sum(
                map(lambda x: torch.count_nonzero(x.grad != 0.0), model.parameters())
            )
            with open(LOGDIR / "updates.csv", "a") as f:
                f.write(f"{epoch},{step},{num_updates}\n")

            ## Apply updates
            if hyperparams.get("fp32_training", False):
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param -= hyperparams["learning_rate"] * param.grad.sign_()
            elif simexp_pool is not None:
                simexp_pool.update_model_params(model, grad_update=True)
            else:
                ## Configure to not apply weak reset pulses to devices we don't want to
                config_sweep_mask(uc_driver, synapse_grads, hyperparams["array_size"])

                # If hardware_mac, currently "MAC" is configured, need to change
                if hyperparams["hardware_mac"]:
                    config_read_write(ARRAY_SIZE, B1530_INIT_DELAY, PATTERN_DEFAULT)

                uc_driver.array_sweep(1)
                memristor_conductances = utils.array_exec_and_fetch(
                    hyperparams["array_size"],
                    current_name=CURRENT_CHANNEL_NAME,
                    voltage_name=VOLTAGE_CHANNEL_NAME,
                )
                memristive_parameters = utils.params_from_conductances(
                    memristor_conductances
                )

                # Once programmed, we reconfigure for performing MACs
                if hyperparams["hardware_mac"]:
                    b1530.dll.clear()
                    b1530.setup([("MAC", PATTERN_DEFAULT)], 1)

                utils.map_memory(
                    MEMORY_MAP,
                    hyperparams["array_offset"],
                    memristive_parameters,
                    model.parameters(),
                    scale=hyperparams["scale_factor"],
                )

                np.save(
                    LOGDIR / "array_conductances" / f"e{epoch}s{step}.npy",
                    memristor_conductances,
                )

            np.savez(
                LOGDIR / "model_params" / f"e{epoch}s{step}.npz",
                **model.state_dict(),
                allow_pickle=False,
            )

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print("Computing Train Accuracy...")
        train_acc = compute_accuracy(model, train_loader)
        print("Computing Test Accuracy...")
        test_acc = compute_accuracy(model, test_loader)
        val_acc = (
            compute_accuracy(model, val_loader)
            if epoch + 1 == hyperparams["num_epochs"]
            else -1
        )
        with open(LOGDIR / "metrics.csv", "a") as f:
            f.write(f"{epoch + 1},{train_loss},{train_acc},{test_acc},{val_acc}\n")

        print(
            f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%, Test Acc: {test_acc * 100:.2f}%"
        )

    print(f"Final Validation Accuracy: {val_acc * 100:.2f}%")

    if simexp_pool is None:
        b1530.dll.clear()
        b1530.dll.closeSession()
        uc_driver.ser.close()
        utils.turn_off_psu()

    return LOGDIR


def run_mac_test(nums=200, mac_pattern="default_sk", read_voltage=0.2):
    from pathlib import Path

    global uc

    path = Path(f"data/mac_tests/{datetime.now().strftime('%d%m_%H%M')}/")
    path.mkdir(parents=True)

    # utils.set_psu("sk")
    uc = uc_sk

    in_features = 32
    out_features = 4

    m = uc.MCDriver(array_size=(out_features, in_features), array_offset=(11, 0))
    m.ack_mode(uc.ACK.FILL_SRs | uc.ACK.SET_CS_EN | uc.ACK.SET_SWEEP_MASK)
    b1530.init()

    print("Reading initial conductances...")
    b1530.setup(
        [("READ", "default_sk")],
        2 * out_features * in_features,
        en_trig=True,
        overwrite_init_delay=175e-6,
    )
    m.array_sweep(0)
    gs = b1530.exec_and_fetch(current_name="VSL0", voltage_name="BL0")[:-1].reshape(
        out_features, 2 * in_features
    )
    np.save(path / "init_conductances.npy", gs, allow_pickle=False)
    weights = 1e6 * (gs[:, ::2] - gs[:, 1::2])

    b1530.dll.clear()
    b1530.setup([("MAC", mac_pattern)], 1)

    model = HardwareLinear(
        in_features,
        out_features,
        m,
        scale_factor=1e6,
        device=device,
        read_voltage=read_voltage,
    )

    print("Doing MACs...")
    random_inputs = np.random.choice([-1, 0, 1], size=(nums, in_features))
    np.save(path / "random_inputs.npy", random_inputs, allow_pickle=False)

    expected_outputs = np.asarray(
        torch.nn.functional.linear(
            utils.to_torch(random_inputs), utils.to_torch(weights)
        )
    )
    np.save(path / "expected_outputs.npy", expected_outputs, allow_pickle=False)

    measured_outputs = np.zeros((len(random_inputs), out_features))
    for i in tqdm(range(nums)):
        out = model(random_inputs[i][None, :]).detach().numpy()
        measured_outputs[i, :] = out

    np.save(path / "measured_outputs.npy", measured_outputs, allow_pickle=False)

    min_ = min(np.min(expected_outputs), np.min(measured_outputs))
    max_ = max(np.max(expected_outputs), np.max(measured_outputs))

    plt.style.use("mystyle")
    plt.figure()
    plt.plot(expected_outputs.ravel(), measured_outputs.ravel(), "o", alpha=0.4)
    plt.plot(np.linspace(min_, max_, 2), np.linspace(min_, max_, 2), "k--", alpha=0.6)
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.gca().set_box_aspect(1)
    plt.xlabel("Expected Output", fontdict={"weight": "bold"})
    plt.ylabel("Measured Output", fontdict={"weight": "bold"})
    plt.savefig(path / "plot.svg", dpi=300, bbox_inches="tight")

    m.ser.close()
    b1530.dll.clear()
    b1530.dll.closeSession()
    utils.turn_off_psu()


def get_devices_from_mmap(path):
    with open(path, "r") as f:
        content = f.read()
    devices = [
        [int(elem[:2]), int(elem[3:6]), b]
        for layer in content.strip().split("=====")
        for elem in layer.strip().replace("\n", " ").split(" ")
        for b in [False, True]
    ]
    return devices


def run_cf_exps():
    import experiments_helpers as eh

    hps = {
        "arch": [32, 48, 48],
        "learning_rule": "cf",
        "chip": "aa",
        "memory_map": "data/memory_maps/arraywr2_32-48-48.dat",
        "array_size": [64, 64],
        "array_offset": [0, 0],
        "scale_factor": [1000, 14000],
        "grad_thresh": [1.73e-05, 0.00012],
        "ff_thetas": [[7.46, 9.08], [6.71, 2.49]],
        "num_epochs": 30,
        "switch_epoch": 20,
        "fp32_training": False,
        "learning_rate": 0.005,
        "hardware_mac": False,
        "ternarize": False,
    }

    devices = get_devices_from_mmap("data/memory_maps/arraywr2_32-48-48.dat")

    for i in range(5):
        print(f"\n\n===== EXP #{i + 1}/5 =====\n\n")
        utils.set_psu("aa")

        uc_driver = uc_aa.MCDriver()
        b1530.init()

        eh.run_experiment(
            uc_driver, "lowvolt_aa_reinit", user_selected_devices=devices
        )

        ## Run Training
        uc_driver.ser.close()
        del uc_driver  # Just make sure it's really closed
        b1530.dll.clear()
        b1530.dll.closeSession()

        sleep(1)

        cur_logdir = run_training(hps)
        run_training(cur_logdir)  # Just double check the accuracies

    utils.turn_off_psu()

    print("===== FINISHED =====")


def run_sff_exps(simexp=False):
    import experiments_helpers as eh

    hps = {
        "arch": [36, 48, 48],
        "learning_rule": "sff",
        "chip": "aa",
        "memory_map": "data/memory_maps/arraywr2_36-48-48.dat",
        "array_size": [64, 64],
        "array_offset": [0, 0],
        "scale_factor": [9500, 2000],
        "grad_thresh": [1.1e-4, 3.2e-3],
        "ff_thetas": [[1.25, 0.08], [6.4, 0.73]],
        "num_epochs": 30,
        "switch_epoch": 10,
        "fp32_training": False,
        "learning_rate": 0.005,
        "hardware_mac": False,
        "ternarize": False,
    }

    if simexp:
        hps["simexp"] = "march_slope_x3_5k"

    devices = get_devices_from_mmap(hps["memory_map"])

    for i in range(5):
        print(f"\n\n===== EXP #{i + 1}/5 =====\n\n")
        utils.set_psu(hps["chip"])

        uc_driver = uc_aa.MCDriver()
        b1530.init()

        # if i != 0:
        eh.run_experiment(uc_driver, "lowvolt_aa_reinit", user_selected_devices=devices)

        ## Run Training
        uc_driver.ser.close()
        del uc_driver  # Just make sure it's really closed
        b1530.dll.clear()
        b1530.dll.closeSession()

        sleep(1)

        cur_logdir = run_training(hps)
        run_training(cur_logdir)  # Just double check the accuracies

    utils.turn_off_psu()

    print("===== FINISHED =====")
