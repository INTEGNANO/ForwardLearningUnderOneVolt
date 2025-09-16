# ruff: noqa: E741
import numpy as np
import pandas as pd

from datetime import datetime
import time
import json
import os
import pathlib
import uuid
from typing import Sequence, Optional, Any


from lvr.constants import WGFMU_names2idxs, FOLDER_PATH
from lvr.drivers import b1530, uc_aa


def store_pattern(pattern: dict) -> str:
    if "name" not in pattern:
        raise ValueError("Pattern has no defined name, cannot create an uuid")

    pattern_uuid = str(uuid.uuid4())

    patterns_path = (
        FOLDER_PATH / "wgfmu_patterns" / pattern["name"] / (pattern_uuid + ".json")
    )
    patterns_path.parent.mkdir(parents=True, exist_ok=True)

    with open(patterns_path, "w") as f:
        json.dump(pattern, f, indent=2)

    return pattern_uuid


def load_pattern(pattern_name: str, pattern_uuid: str) -> dict:
    patterns_path = (
        FOLDER_PATH / "wgfmu_patterns" / pattern_name / (pattern_uuid + ".json")
    )

    with open(patterns_path, "r") as f:
        return json.load(f)


def load_patterns_of_sequence(sequence):
    patterns = {}
    for pattern, pattern_uuid, _ in sequence:
        pattern = pattern.upper()
        if pattern != "SLEEP":
            patterns[(pattern, pattern_uuid)] = load_pattern(pattern, pattern_uuid)
    return patterns


def create_read_pattern(
    read_pulse_length,
    read_voltage,
    read_voltage_gate,
    read_range,
    time_init_delay_read,
    trf,
    name=None,
):
    time_pattern_read = [
        time_init_delay_read,
        trf,
        read_pulse_length,
        trf,
        time_init_delay_read,
    ]
    vPatGND = [0.0, 0.0, 0.0, 0.0, 0.0]
    vPatI = [0.0, 0.0, 0.0, 0.0, 0.0]
    vPatVcol = [0.0, read_voltage, read_voltage, 0.0, 0.0]
    vPatVg = [0.0, read_voltage_gate, read_voltage_gate, 0.0, 0.0]

    init_delay = time_init_delay_read + read_pulse_length * 0.25
    sample_interval = sum(time_pattern_read)  # Dont care, taking single data point
    meas_average_time = read_pulse_length * 0.5

    patterns = {
        "Vcol": {
            "id": WGFMU_names2idxs["Vcol"],
            "name": "vcol",
            "voltage_pattern": vPatVcol,
            "time_pattern": time_pattern_read,
            "meas": "voltage",
            "range": "5V",
            "sample_count": 1,
        },
        "I": {
            "id": WGFMU_names2idxs["I"],
            "name": "iI",
            "voltage_pattern": vPatI,
            "time_pattern": time_pattern_read,
            "meas": "current",
            "range": read_range,
            "sample_count": 1,
        },
        "GND": {
            "id": WGFMU_names2idxs["GND"],
            "name": "gnd",
            "voltage_pattern": vPatGND,
            "time_pattern": time_pattern_read,
            "meas": "voltage",
            "range": "5V",
            "sample_count": 1,
        },
        "Vg": {
            "id": WGFMU_names2idxs["Vg"],
            "name": "vg",
            "voltage_pattern": vPatVg,
            "time_pattern": time_pattern_read,
            "meas": "voltage",
            "range": "5V",
            "sample_count": 1,
        },
    }

    return dict(
        patterns=patterns,
        meas_average_time=meas_average_time,
        init_delay=init_delay,
        sample_interval=sample_interval,
        name=name,
    )


def create_write_pattern(
    pulse_length,
    voltage,
    voltage_gate,
    time_init_delay,
    gate_preopen_delay,
    trf,
    name=None,
):
    # WRITE
    time_pattern = [
        time_init_delay,
        trf,
        gate_preopen_delay,
        trf,
        pulse_length,
        trf,
        gate_preopen_delay,
        trf,
        time_init_delay,
    ]
    vPatGND = [0] * len(time_pattern)
    vPatI = [0] * len(time_pattern)
    vPatVg = [
        0,
        voltage_gate,
        voltage_gate,
        voltage_gate,
        voltage_gate,
        voltage_gate,
        voltage_gate,
        0,
        0,
    ]
    vPatVcol = [0, 0, 0, voltage, voltage, 0, 0, 0, 0]

    patterns = {
        "Vcol": {
            "id": WGFMU_names2idxs["Vcol"],
            "name": "vcol",
            "voltage_pattern": vPatVcol,
            "time_pattern": time_pattern,
            "meas": "voltage",
            "range": "5V",
            "sample_count": 1,
        },
        "I": {
            "id": WGFMU_names2idxs["I"],
            "name": "iI",
            "voltage_pattern": vPatI,
            "time_pattern": time_pattern,
            "meas": "none",
        },
        "GND": {
            "id": WGFMU_names2idxs["GND"],
            "name": "gnd",
            "voltage_pattern": vPatGND,
            "time_pattern": time_pattern,
            "meas": "none",
        },
        "Vg": {
            "id": WGFMU_names2idxs["Vg"],
            "name": "vg",
            "voltage_pattern": vPatVg,
            "time_pattern": time_pattern,
            "meas": "none",
        },
    }

    return dict(
        patterns=patterns,
        name=name,
    )


def create_rw_pattern(
    read_pulse_length,
    write_pulse_length,
    read_voltage,
    write_voltage,
    read_voltage_gate,
    write_voltage_gate,
    read_range,
    write_range,
    time_init_delay_read,
    time_init_delay_write,
    gate_preopen_delay_write,
    trf,
    name=None,
):
    # WRITE
    time_pattern_write = [
        time_init_delay_write,
        trf,
        gate_preopen_delay_write,
        trf,
        write_pulse_length,
        trf,
        gate_preopen_delay_write,
        trf,
        # time_init_delay_write,  # Do not delay here as this pulse will be appended to the read pulse
    ]
    vPatGNDWrite = [0] * len(time_pattern_write)
    vPatIWrite = [0] * len(time_pattern_write)
    vPatVgWrite = [
        0,
        write_voltage_gate,
        write_voltage_gate,
        write_voltage_gate,
        write_voltage_gate,
        write_voltage_gate,
        write_voltage_gate,
        0,
    ]
    vPatVcolWrite = [0, 0, 0, write_voltage, write_voltage, 0, 0, 0]

    # READ
    time_pattern_read = [
        time_init_delay_read,
        trf,
        read_pulse_length,
        trf,
        gate_preopen_delay_write,
    ]
    vPatGNDRead = [0] * len(time_pattern_read)
    vPatIRead = [0] * len(time_pattern_read)
    vPatVgRead = [0.0, read_voltage_gate, read_voltage_gate, 0.0, 0.0]
    vPatVcolRead = [0.0, read_voltage, read_voltage, 0.0, 0.0]

    # Append Read to Write
    time_pattern = time_pattern_write + time_pattern_read
    vPatGND = vPatGNDWrite + vPatGNDRead
    vPatI = vPatIWrite + vPatIRead
    vPatVcol = vPatVcolWrite + vPatVcolRead
    vPatVg = vPatVgWrite + vPatVgRead

    init_delay = (
        sum(time_pattern_write) + time_init_delay_read + read_pulse_length * 0.25
    )
    sample_interval = sum(time_pattern)  # Dont care, taking single data point
    meas_average_time = read_pulse_length * 0.5

    range_change_time = [
        sum(time_pattern_write),  # Current measurement range for Write
        sum(time_pattern_write)
        + sum(time_pattern_read)
        - gate_preopen_delay_write,  # & for Read
    ]

    patterns = {
        "BL0": {
            "id": WGFMU_names2idxs["BL0"],
            "name": "bl",
            "voltage_pattern": vPatVcol,
            "time_pattern": time_pattern,
            "meas": "voltage",
            "range": "5V",
            "sample_count": 1,
        },
        "HSL0": {
            "id": WGFMU_names2idxs["HSL0"],
            "name": "hsl",
            "voltage_pattern": vPatI,
            "time_pattern": time_pattern,
            "meas": "current",
            "range": write_range,
            "write_range": write_range,
            "read_range": read_range,
            "sample_count": 1,
        },
        "VSL0": {
            "id": WGFMU_names2idxs["VSL0"],
            "name": "vsl",
            "voltage_pattern": vPatGND,
            "time_pattern": time_pattern,
            "meas": "voltage",
            "range": "5V",
            "sample_count": 1,
        },
        "WL0": {
            "id": WGFMU_names2idxs["WL0"],
            "name": "wl",
            "voltage_pattern": vPatVg,
            "time_pattern": time_pattern,
            "meas": "voltage",
            "range": "5V",
            "sample_count": 1,
        },
    }

    return dict(
        patterns=patterns,
        meas_average_time=meas_average_time,
        init_delay=init_delay,
        sample_interval=sample_interval,
        range_change_time=range_change_time,
        name=name,
    )


def get_device_name(row: int, col: int, bar: bool) -> str:
    return f"R{row:02d}_{col:02d}" + ("_bar" if bar else "")


def name_to_tuple(name: str) -> tuple[int, int, bool]:
    """/!\ May SILENTLY give wrong result if not in the right format `RXX_XX[_bar]`"""
    return int(name[1:3]), int(name[4:6]), len(name) > 6


def get_device_logfile(device_name: str) -> pathlib.Path:
    return FOLDER_PATH / "devices" / (device_name + ".csv")


def get_formed_devices() -> tuple[
    list[tuple[int, int, bool]], list[tuple[int, int, bool]]
]:
    formed_devices, unformed_devices = [], []
    for row in range(0, 64):
        for col in range(0, 64):
            for bar in [True, False]:
                dev_name = get_device_name(row, col, bar)

                if os.path.exists(FOLDER_PATH / "devices" / (dev_name + ".csv")):
                    formed_devices.append((row, col, bar))
                else:
                    unformed_devices.append((row, col, bar))

    return formed_devices, unformed_devices


def select_formed_devices(forming_uuid: str) -> list[tuple[int, int, bool]]:
    formed_devices = []
    for row in range(0, 64):
        for col in range(0, 64):
            for bar in [True, False]:
                dev_name = get_device_name(row, col, bar)
                dev_logfile = get_device_logfile(dev_name)

                if os.path.exists(dev_logfile):
                    df = pd.read_csv(dev_logfile)
                    if (
                        df["pattern"] == "FORMING"
                        and df["pattern_uuid"] == forming_uuid
                    ):
                        formed_devices.append((row, col, bar))

    return formed_devices


def select_unique_devices(count: int) -> tuple[list[tuple[int, int, bool]], str]:
    devices: list[tuple[int, int, bool]] = []
    while len(devices) < count:
        r: int = np.random.randint(low=0, high=64)
        c: int = np.random.randint(low=0, high=64)
        b = bool(np.random.randint(2))

        dev_name = get_device_name(r, c, b)
        if not os.path.exists(FOLDER_PATH.joinpath("devices").joinpath(dev_name)):
            if (r, c, b) not in devices:
                devices.append((r, c, b))

    name = datetime.now().strftime("%d_%m_%Y-%H_%M_%S") + ".csv"
    with open(FOLDER_PATH.joinpath("device_selection").joinpath(name), "w") as f:
        f.write("row,column,bar\n")
        for i in range(count):
            f.write(
                str(devices[i][0])
                + ","
                + str(devices[i][1])
                + ","
                + str(devices[i][2])
                + "\n"
            )

    return devices, name


def create_experiment(
    selected_devices: list[tuple[int, int, bool]],
    sequence: Sequence[tuple[str, str, Optional[str]]],
    repeat_args: Optional[dict[int, dict[str, Any]]] = None,
    forming_args: Optional[dict[str, Any]] = None,
) -> str:
    exp_uuid = str(uuid.uuid4())
    exp_path = FOLDER_PATH.joinpath("experiments").joinpath(exp_uuid)

    FOLDER_PATH.joinpath("devices").mkdir(parents=True, exist_ok=True)

    for r, c, b in selected_devices:
        device_name = get_device_name(r, c, b)
        device_log_file = FOLDER_PATH.joinpath("devices").joinpath(device_name + ".csv")

        # Create device log file if does not exist
        if not device_log_file.is_file():
            with open(device_log_file, "w") as f:
                f.write(
                    "exp_uuid,date,operation,pattern,pattern_uuid,mode,i,time,voltage,current,resistance\n"
                )

    # Validate experiment
    weakreset_id = None
    for i, (pattern, pattern_uuid, operation) in enumerate(sequence):
        pattern = pattern.upper()

        if pattern == "SLEEP":
            continue

        if (
            not FOLDER_PATH.joinpath("wgfmu_patterns")
            .joinpath(pattern)
            .joinpath(pattern_uuid + ".json")
            .exists()
        ):
            raise ValueError(
                f"`create_experiment`: Use of a non-existing pattern {pattern} {pattern_uuid}"
            )

        pattern = pattern.upper()

        if pattern == "FORMING":
            if forming_args is not None and (
                "max_Vcol" in forming_args and "Vcol_step" in forming_args
            ):
                raise NotImplementedError(
                    "`create_experiment`: Incremental `FORMING` pulse not implemented."
                )

        # if pattern == "WEAK_RESET":
        #     if weakreset_id is not None:
        #         raise ValueError(
        #             "`create_experiment`: At most one `WEAK_RESET` pattern can be set in a given sequence."
        #         )
        #     weakreset_id = i
        if pattern not in ["READ", "WEAK_RESET", "HARD_SET", "HARD_RESET", "FORMING"]:
            raise ValueError(
                f"`create_experiment`: The sequence contains an unknown pattern: `{pattern}"
            )

    exp_path.mkdir(parents=True)

    with open(exp_path / "params.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().strftime("%d_%m_%Y-%H_%M_%S"),
                "selected_devices": selected_devices,
                "sequence": [(s[0].upper(), s[1], s[2]) for s in sequence],
                "repeat_args": repeat_args or {},
                "forming_args": forming_args or {},
            },
            f,
            indent=2,
        )

    return exp_uuid


def run_experiment(
    uc_driver: uc_aa.MCDriver,
    exp_uuid: str,
    user_selected_devices: Optional[list[tuple[(int, int, bool)]]] = None,
) -> Optional[str]:
    exp_path = FOLDER_PATH / "experiments" / exp_uuid

    with open(exp_path / "params.json", "r") as f:
        exp_params = json.load(f)

    selected_devices = exp_params["selected_devices"]
    sequence = exp_params["sequence"]
    repeat_args = {int(k): v for k, v in exp_params["repeat_args"].items()}
    forming_args: dict[str, Any] = exp_params["forming_args"]

    if user_selected_devices is not None:
        if not all([u_dev in selected_devices] for u_dev in user_selected_devices):
            raise ValueError(
                "`run_experiment`: `user_selected_devices` must be a subset of the experiment `selected_devices`. Nothing has been run before raising this, no worries."
            )
        selected_devices = user_selected_devices

    executed_experiment = exp_params.get("executed", False)
    if executed_experiment:
        sister_uuid = create_experiment(
            selected_devices, sequence, repeat_args, forming_args
        )
        if "sister_experiments" not in exp_params:
            exp_params["sister_experiments"] = []
        exp_params["sister_experiments"].append(sister_uuid)
        with open(exp_path.joinpath("params.json"), "w") as f:
            json.dump(exp_params, f, indent=2)

        run_experiment(uc_driver, sister_uuid)
        return sister_uuid

    patterns = load_patterns_of_sequence(sequence)

    def _setup_b1530(config, num, prefix=""):
        b1530.dll.clear()
        modes, ranges = [0, 0, 0, 0], [0, 0, 0, 0]
        measure = False
        for v in config["patterns"].values():
            i = v["id"] - 1

            if "range" in v:
                measure = True
                modes[i] = v["meas"]
                ranges[i] = v["range"]
            else:
                modes[i] = "voltage"

        en_trigs = [False, False, False, False]
        b1530.setup_wgfmus(modes, ranges, en_trigs, measure=measure)
        b1530.init_patterns(config, prefix)

        for pattern_name, pattern in config["patterns"].items():
            b1530.dll.addSequence(
                b1530.WGFMU_channels[pattern["id"]], prefix + "_" + pattern_name, num
            )

    def exec_and_log(config, pulse_count):
        _setup_b1530(config, pulse_count)
        T, V, I, R = b1530.exec_and_fetch(return_all=True)

        if np.any(R < 0):
            print(
                "[WARNING] Invalid measurement for current device (negative resistance)"
            )

        log_info = "\n".join(
            [
                ",".join(
                    map(
                        str,
                        [
                            exp_uuid,
                            datetime.now().strftime("%d_%m_%Y-%H_%M_%S"),
                            operation or "-",
                            pattern,
                            pattern_uuid,
                            "-",
                            k,
                            T[k],
                            V[k],
                            I[k],
                            R[k],
                        ],
                    )
                )
                for k in range(len(V))
            ]
        )

        with open(device_log_file, "a") as f:
            f.write(log_info + "\n")

        return R

    for r, c, b in selected_devices:
        global repeat_counter  # Berk
        repeat_counter = {k: 0 for k in repeat_args.keys()}
        R = None

        def stop_repeat_condition(i: int, R: Optional[np.ndarray]) -> bool:
            global repeat_counter

            if i not in repeat_args or (
                ("LRS_threshold" not in repeat_args[i])
                and ("max_repeat_count" not in repeat_args[i])
            ):
                return R is None  # Will be true at first iteration, then false

            repeat = True
            if "LRS_threshold" in repeat_args[i]:
                repeat = repeat and (R is None or R > repeat_args[i]["LRS_threshold"])
            if "max_repeat_count" in repeat_args[i]:
                repeat = (
                    repeat and repeat_counter[i] < repeat_args[i]["max_repeat_count"]
                )
            repeat_counter[i] += 1
            return repeat

        device_name = get_device_name(r, c, b)
        device_log_file = FOLDER_PATH / "devices" / (device_name + ".csv")

        for i, (pattern, pattern_uuid, operation) in enumerate(sequence):
            pattern = pattern.upper()
            pulse_count = (
                repeat_args[i].get("pulse_counts", 1) if i in repeat_args else 1
            )

            print(
                f"[INFO] [{device_name}] Executing pattern `{pattern}`for {pulse_count} pulse(s)"
            )

            if pattern != "SLEEP":
                config = patterns[(pattern, pattern_uuid)]

            if pattern == "FORMING":
                # Vcol_form_temp = config["patterns"]["Vcol"]["voltage_pattern"][3]

                uc_driver.configure_sr(col=c, row=r, bar=b, set=True)
                while stop_repeat_condition(i, R):
                    R = exec_and_log(config, pulse_count)

                    # if "max_Vcol" in forming_args and "Vcol_step" in forming_args:
                    #     if Vcol_form_temp < forming_args["max_Vcol"]:
                    #         Vcol_form_temp += forming_args["Vcol_step"]
            elif pattern == "SLEEP":
                time.sleep(
                    float(pattern_uuid)
                )  # Hack to use this element as a param rather than actual uuid
            elif pattern == "READ":
                uc_driver.configure_sr(col=c, row=r, bar=b, set=True)
                exec_and_log(config, pulse_count)
            elif pattern == "HARD_SET":
                R = None
                uc_driver.configure_sr(col=c, row=r, bar=b, set=True)
                while stop_repeat_condition(i, R):
                    R = exec_and_log(config, pulse_count)
            elif pattern == "HARD_RESET":
                uc_driver.configure_sr(col=c, row=r, bar=b, set=False)
                exec_and_log(config, pulse_count)
            elif pattern == "WEAK_RESET":
                uc_driver.configure_sr(col=c, row=r, bar=b, set=False)
                R = exec_and_log(config, pulse_count)

    exp_params["executed"] = True
    with open(exp_path / "params.json", "w") as f:
        json.dump(exp_params, f, indent=2)


def get_weakreset_exp_selected_devices(exp_uuid: str) -> list[tuple[int, int, bool]]:
    exp_path = FOLDER_PATH / "experiments" / exp_uuid

    with open(exp_path / "weak_resets.json", "r") as f:
        weakresets = json.load(f)

    return [name_to_tuple(dev) for dev in weakresets["pulses"].keys()]


def restart_weakreset_exp(uc_driver: uc_aa.MCDriver, exp_uuid: str) -> Optional[str]:
    exp_path = FOLDER_PATH / "experiments" / exp_uuid

    with open(exp_path / "params.json", "r") as f:
        exp_params = json.load(f)
    if "executed" in exp_params and exp_params["executed"]:
        print(
            f"[INFO] `restart_weakreset_exp`: Experiment `{exp_uuid}` already completed."
        )
        return

    def has_been_formed(dev_name: str) -> bool:
        with open(FOLDER_PATH / "devices" / (dev_name + ".csv"), "r") as f:
            f.readline()
            return len(f.readline()) != 0

    not_executed_devices = [
        dev
        for dev in exp_params["selected_devices"]
        if not has_been_formed(get_device_name(*dev))
    ]

    confirm = input(
        f"[?] Restart weak reset for {len(not_executed_devices)} devices, between {get_device_name(*not_executed_devices[0])} and {get_device_name(*not_executed_devices[-1])}"
    )
    if confirm.upper() in ["Y", "YES", "OUI"]:
        print(f"[INFO] Running again experiment  {exp_uuid} ")
        return run_experiment(uc_driver, exp_uuid, not_executed_devices)
    else:
        print("Aborted.")
