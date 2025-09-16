import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from lvr.drivers import b1530
from lvr.drivers import uc_sk as uc
from lvr.constants import WGFMU_channels, WGFMU_names2idxs

from typing import Optional
from time import sleep


class HardwareLinear(Module):
    __constants__ = [
        "in_features",
        "out_features",
        "scale_factor",
        "read_voltage",
        "device",
    ]
    in_features: int
    out_features: int
    weight: Tensor  # Dummy, just to store grads

    uc_driver: uc.MCDriver
    scale_factor: float
    read_voltage: float

    saved_input: Optional[Tensor]

    device: torch.device

    def __init__(
        self,
        in_features: int,
        out_features: int,
        uc_driver: uc.MCDriver,
        scale_factor: float,
        device: torch.device,
        read_voltage=0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.zeros(out_features, in_features, device=device)
        )
        self.uc_driver = uc_driver
        self.scale_factor = scale_factor
        self.read_voltage = read_voltage
        self.saved_input = None
        self.device = device

        if self.out_features > self.uc_driver.array_size[0]:
            raise ValueError(
                f"Output features ({self.out_features}) exceed the hardware array height ({self.uc_driver.array_size[0]})"
            )

        self.uc_driver.ack_mode(
            uc.ACK.FILL_SRs | uc.ACK.SET_CS_EN | uc.ACK.SET_SWEEP_MASK
        )

    def forward(self, input: Tensor) -> Tensor:
        self.uc_driver.set_cs_en(0b10101)  # Activate BLs and Horizontal WLs/SLs for MAC

        self.saved_input = input  # For custom backward later

        output = torch.zeros((input.shape[0], self.out_features), device=self.device)
        for batch_idx in range(input.shape[0]):
            inp = input[batch_idx]
            for row in range(self.out_features):
                self._configure_sr_mac(inp, row + self.uc_driver.array_offset[0])

                b1530.dll.execute()
                b1530.dll.waitUntilCompleted()

                mac_current = -b1530.dll.getMeasureValues(
                    WGFMU_channels[WGFMU_names2idxs["HSL1"]], 0, 1
                )[2][0]

                mac_current *= self.scale_factor / self.read_voltage  # Vread ~= 0.2V
                output[batch_idx, row] = mac_current

        self.uc_driver.set_cs_en(
            0b11010
        )  # Re-Activate BLs and Vertical WLs/SLs for MAC
        return output

    def _configure_sr_mac(self, input: Tensor, row: int):
        assert input.shape == (32,), "Inputs must be a 32-element array"

        # Indices are the same as SR.HWL/HSL/...
        sr_words = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ]

        sr_words[uc.SR.HWL] = (0, 1 << (row * 2))
        sr_words[uc.SR.HSL] = (0, 1 << (row * 3 + 2))

        # BL is a 2* 32*2bit shift register
        # BL[i] = 0b00: connected to GND
        # BL[i] = 0b01: connected to BL0 = 0.5V
        # BL[i] = 0b10: connected to BL1 = 0.9V
        # BL[i] = 0b11: connected to V0=BL2 = 0.7V

        # Within the shit registers, the LSB is at index 0
        # so to encode 0b01, 0b10 must actually be put into the SR

        # Because HSL will be forced to V0,
        # To encode +1, we need BL[i] = BL1
        # To encode -1, we need BL[i] = BL0
        # To encode 0, we need BL[i] = V0

        # Because the memristors are used in differential pair, BL[2*i+1] = -BL[2*i]
        for col in range(32):
            word_pos = int(col < 16)  # 0 for least 64 bits, 1 for most 64 bits
            if input[col] > 0:
                sr_words[uc.SR.BL][word_pos] |= 0b10 << (
                    (4 * col + 0) % 64
                )  # memr -> BL1
                sr_words[uc.SR.BL][word_pos] |= 0b01 << (
                    (4 * col + 2) % 64
                )  # comp -> BL0
            elif input[col] < 0:
                sr_words[uc.SR.BL][word_pos] |= 0b01 << (
                    (4 * col + 0) % 64
                )  # memr -> BL0
                sr_words[uc.SR.BL][word_pos] |= 0b10 << (
                    (4 * col + 2) % 64
                )  # comp -> BL1
            else:
                sr_words[uc.SR.BL][word_pos] |= 0b11 << (
                    (4 * col + 0) % 64
                )  # memr -> V0
                sr_words[uc.SR.BL][word_pos] |= 0b11 << (
                    (4 * col + 2) % 64
                )  # comp -> V0

        self.uc_driver.configure_sr_words(sr_words)

        return sr_words

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"
