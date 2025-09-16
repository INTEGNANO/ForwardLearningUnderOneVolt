from pathlib import Path

## Connections Awesome Array
# WGFMU1: Vin1Row
# WGFMU2: Vin1Col
# WGFMU3: Vin2Row
# WGFMU4: Vin2Col

## Connections for Swiss Knife
# WGFMU1: WL0
# WGFMU2: VSL0
# WGFMU3: HSL1
# WGFMU4: BL0
# PSU Chan2: BL1

WGFMU_names2idxs = {
    ## Awesome Array
    "GND": 1,
    "I": 2,
    "Vg": 3,
    "Vcol": 4,
    ## Swiss Knife
    "WL0": 1,  # Connected to both HWL0 and VWL0
    "VSL0": 2,
    "BL0": 4,
    "HSL1": 3,
}

WGFMU_channels = {
    1: 101,
    2: 102,
    3: 201,
    4: 202,
}

FOLDER_PATH = Path("data")
