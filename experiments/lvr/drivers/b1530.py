import numpy as np

import B1530driver as dll
from lvr.constants import WGFMU_channels, WGFMU_names2idxs

from typing import Union
from copy import deepcopy
import json


def setup_wgfmus(
    modes: list[str], ranges: list[str], en_trigs: list[bool], *, measure=True
):
    for channel, mode, range, en_trig in zip(
        WGFMU_channels.values(), modes, ranges, en_trigs
    ):
        if measure:
            dll.setOperationMode(channel, dll._operationMode["fastiv"])
            dll.setMeasureMode(channel, dll._measureMode[mode])
        else:
            dll.setOperationMode(channel, dll._operationMode["pg"])

        if measure:
            if mode == "current":
                dll.setMeasureCurrentRange(channel, dll._measureCurrentRange[range])
            elif mode == "voltage":
                dll.setForceVoltageRange(channel, dll._forceVoltageRange["auto"])
                dll.setMeasureVoltageRange(channel, dll._measureVoltageRange[range])
        elif mode == "voltage":
            dll.setForceVoltageRange(channel, dll._forceVoltageRange["auto"])

        if en_trig:
            dll.setTriggerOutMode(
                channel,
                dll._triggerOutMode["event"],
                dll._triggerOutPolarity["positive"],
            )


def init_patterns(config, prefix="", overwrite_init_delay: Union[float, None] = None):
    for pattern_name, pattern in config["patterns"].items():
        pattern_name = prefix + "_" + pattern_name

        time_pattern = deepcopy(pattern["time_pattern"])

        if overwrite_init_delay is not None:
            time_pattern[0] = overwrite_init_delay

        dll.createPattern(pattern_name, pattern["voltage_pattern"][0])
        dll.addVectors(
            pattern_name,
            time_pattern,
            pattern["voltage_pattern"],
            len(pattern["voltage_pattern"]),
        )

        if pattern["meas"] == "none":
            continue

        if overwrite_init_delay is not None:
            init_delay = (
                config["init_delay"] - pattern["time_pattern"][0] + overwrite_init_delay
            )
        else:
            init_delay = config["init_delay"]

        dll.setMeasureEvent(
            pattern_name,
            pattern_name + "_MEAS",
            init_delay,
            pattern["sample_count"],
            config["sample_interval"],
            config["meas_average_time"],
            dll._measureEventData["averaged"],
        )

        if (
            "range_change_time" in config
            and "read_range" in pattern
            and "write_range" in pattern
        ):
            dll.setRangeEvent(
                pattern_name,
                pattern_name + "_RANGE-w2r",
                config["range_change_time"][0],
                dll._measureCurrentRange[pattern["read_range"]],
            )
            dll.setRangeEvent(
                pattern_name,
                pattern_name + "_RANGE-r2w",
                config["range_change_time"][1],
                dll._measureCurrentRange[pattern["write_range"]],
            )


def init():
    dll.openSession("GPIB1::18::INSTR")
    dll.initialize()
    dll.connect(101)
    dll.connect(102)
    dll.connect(201)
    dll.connect(202)


def setup(configs, num, *, en_trig=False, overwrite_init_delay=None):
    configs_json = []

    for prefix, uuid in configs:
        with open(f"data/wgfmu_patterns/{prefix}/{uuid}.json", "r") as f:
            configs_json.append(json.load(f))

    modes, ranges = [0, 0, 0, 0], [0, 0, 0, 0]
    for v in configs_json[-1]["patterns"].values():
        i = v["id"] - 1
        modes[i] = v["meas"]
        ranges[i] = v["range"]
    en_trigs = [False, False, en_trig, False]
    setup_wgfmus(modes, ranges, en_trigs)

    # Create Patterns
    for i, c in enumerate(configs_json):
        prefix = configs[i][0]

        init_patterns(c, prefix, overwrite_init_delay)

        for pattern_name, pattern in c["patterns"].items():
            if pattern["id"] == 3:  # Trigger on chan 3
                trig_chan_name = pattern_name
                break

        if en_trig:
            dll.setTriggerOutEvent(
                f"{prefix}_{trig_chan_name}", f"{prefix}_{trig_chan_name}_TRIG", 0, 2e-6
            )

    # Create Sequences
    for i, c in enumerate(configs_json):
        prefix = configs[i][0]

        # When triggering, the very last sequence must have an additional dummy pulse to end the sweep
        if en_trig and i + 1 == len(configs_json):
            num += 1

        for pattern_name, pattern in c["patterns"].items():
            dll.addSequence(
                WGFMU_channels[pattern["id"]], prefix + "_" + pattern_name, num
            )


def exec_and_fetch(return_all=False, *, current_name="I", voltage_name="Vcol"):
    """If return_all=True, returns t, v, i, r ; otherwise, returns g"""
    dll.execute()
    dll.waitUntilCompleted()

    _, _, iI_datapoints_count = dll.getMeasureValueSize(
        WGFMU_channels[WGFMU_names2idxs[current_name]]
    )

    _, _, Vcol_datapoints_count = dll.getMeasureValueSize(
        WGFMU_channels[WGFMU_names2idxs[voltage_name]]
    )

    _, _, iI = dll.getMeasureValues(
        WGFMU_channels[WGFMU_names2idxs[current_name]],
        0,
        iI_datapoints_count,
    )
    _, t, vcol = dll.getMeasureValues(
        WGFMU_channels[WGFMU_names2idxs[voltage_name]],
        0,
        Vcol_datapoints_count,
    )

    iI = np.asarray(iI)
    t = np.asarray(t)
    vcol = np.asarray(vcol)

    if return_all:
        return t, vcol, iI, np.abs(vcol / iI)
    else:
        return np.abs(iI / vcol)
