import torch

import h5py
import os
from math import prod, ceil
from typing import Sequence

class Pool:
    def __init__(self, arch: Sequence[int], pool_filename: str, scale_factor: Sequence[int] | int):
        self.layer_sizes = list(zip(arch[1:], arch[:-1]))

        if not isinstance(scale_factor, Sequence):
            self.scale_factor = [scale_factor] * len(self.layer_sizes)
        elif len(scale_factor) != len(self.layer_sizes):
            raise ValueError(f"Number of elements in per-layer scale factor ({len(scale_factor)}) does not match number of layers ({len(self.layer_sizes)})")
        else:
            self.scale_factor = scale_factor
        with h5py.File(rf"{os.environ['DATA']}\WeakResets\{pool_filename}.hdf5") as f:
            self.devs = 1. / torch.Tensor(list(f.values())) # stored in resistance -> convert to conductances

        pool_size = self.devs.size(0)

        self.param_idxs = [
            torch.zeros((*layer_size, 2, 2), dtype=int)
            for layer_size in self.layer_sizes
        ]

        for i in range(len(self.layer_sizes)):
            total_devices = 2 * prod(self.layer_sizes[i])
            devices_order = torch.concat([torch.randperm(pool_size) for _ in range(ceil(total_devices / pool_size))])[:total_devices]
            self.param_idxs[i][..., 0] = devices_order.reshape(*self.layer_sizes[i], 2)


    def update_model_params(self, model, *, grad_update=False):
        with torch.no_grad():
            for i, (param, param_idx) in enumerate(zip(model.parameters(), self.param_idxs)):
                if grad_update and param.grad is not None:
                    param_idx[..., 0, 1] += param.grad > 0 # Pos grad = neg update = we decr the conductance of g+
                    param_idx[..., 1, 1] += param.grad < 0 # Neg grad = pos update = we decr the conductance of g-
                
                param_pos = self.devs[param_idx[..., 0, 0], param_idx[..., 0, 1]]
                param_neg = self.devs[param_idx[..., 1, 0], param_idx[..., 1, 1]]

                param.copy_(self.scale_factor[i] * (param_pos - param_neg))
