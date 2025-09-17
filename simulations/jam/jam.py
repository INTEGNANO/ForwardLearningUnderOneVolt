from __future__ import annotations

import h5py
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand

import collections.abc as abc
from pathlib import Path
import math

from typing import NamedTuple, Union, Optional, Any
from numpy.typing import NDArray

import copy

### UTILS ###
def is_weight_wrapper(x: object) -> bool:
    """An object `x` is a weight wrapper if it has a 'weight' property"""
    return hasattr(x, "weight") and isinstance(type(x).weight, property)


def eval_weight(weight):
    if isinstance(weight, jax.Array):
        return weight
    elif is_weight_wrapper(weight):
        return weight.weight
    else:
        return ValueError(
            f"Weight of type {type(weight)} is neither an array nor a weight wrapper, cannot evaluate"
        )


def map_params(f: abc.Callable, params: jax.Array, *rest) -> jax.Array:
    return jax.tree.map(f, params, *rest, is_leaf=is_weight_wrapper)


def eval_params(params):
    return map_params(eval_weight, params)


# Make a Memristor Location dict into a formatted string
def get_memristor_name(mem_loc: dict[str, Any]) -> str:
    return f"R{mem_loc['row']:02d}_{mem_loc['col']:02d}" + (
        "_bar" if mem_loc["bar"] else ""
    )


### DEVICE POOL ###
@jax.tree_util.register_static
class DevicePool:
    def __init__(
        self,
        filename: Union[Path, str],
        key: jax.Array,
        devices_count: Optional[int] = None,
        devices_select: Optional[abc.Callable] = None,
    ):
        self._key = key
        self._filename = Path(filename)

        with h5py.File(self._filename, "r") as f:
            h5_devices = list(f.values())

            if devices_count is not None:
                if devices_count <= len(h5_devices):
                    h5_devices = h5_devices[:devices_count]
                else:
                    print(
                        f"[WARN] Initialized DevicePool with {devices_count} devices, but only {len(h5_devices)} in file '{filename}'"
                    )

            if devices_select is not None:
                device_filter = devices_select(np.asarray(h5_devices)).astype(int)
                self.devices: jax.Array = jnp.asarray(h5_devices)[device_filter]
            else:
                device_filter = None
                self.devices: jax.Array = jnp.asarray(h5_devices)[:, :]

            self._devices_metadata = np.array(
                [
                    dict(d.attrs.items())
                    for i, d in enumerate(h5_devices)
                    if device_filter is None or i in device_filter
                ]
            )

        if self.devices.ndim != 2:
            raise ValueError(
                f"Invalid devices shape, expected 2 dimensions, got {self.devices.ndim}"
            )

        self.num_devices, self.num_measurement = self.devices.shape

        self.request_metadatas: dict[
            str | int, tuple[NDArray[np.uint32], NDArray[Any]]
        ] = dict()
        self._request_anonym_index: int = 0

        self._devices_order = np.arange(self.num_devices, dtype=np.uint32)
        self._shuffle_devices_order()

        self.pool_index = 0

    def _shuffle_devices_order(self) -> NDArray[np.uint32]:
        self._key, permutation_key = jrand.split(self._key)
        permut = jax.random.permutation(
            permutation_key, self._devices_order, independent=True
        )
        self._devices_order = self._devices_order[permut]

        return self._devices_order

    def request_couple(
        self, shape: Union[abc.Sequence[int], int] = (1,), uid: Optional[str] = None
    ) -> jax.Array:
        if not isinstance(shape, abc.Sequence):
            shape = (shape,)

        total_req_size = 2 * math.prod(shape)  # *2 for pos and neg
        devices_left_in_pool = len(self._devices_order) - self.pool_index

        if total_req_size > devices_left_in_pool:
            print(
                "[WARN] Out-of-device: reshuffling the pool for request of shape", shape
            )

            sel_dev_chunks = [self._devices_order[self.pool_index :]]
            total_req_size -= devices_left_in_pool  # We empty the pool
            self._shuffle_devices_order()

            while total_req_size > len(self._devices_order):
                sel_dev_chunks.append(self._devices_order[:])
                total_req_size -= len(self._devices_order)

                self._shuffle_devices_order()

            self.pool_index = total_req_size
            sel_dev_chunks.append(self._devices_order[: self.pool_index])

            selected_devices = np.concatenate(sel_dev_chunks)
        else:
            selected_devices = self._devices_order[
                self.pool_index : self.pool_index + total_req_size
            ]
            self.pool_index += total_req_size

        if uid is None:
            id = self._request_anonym_index
            self._request_anonym_index += 1
        else:
            id = uid

        self.request_metadatas[id] = (
            selected_devices,
            self._devices_metadata[selected_devices],
        )
        return jax.device_put(selected_devices.reshape((2, *shape)))


### MEMRISTOR WEIGHT ###
class MemristorWeight(NamedTuple):
    pos_pulse_id: jax.Array
    neg_pulse_id: jax.Array
    statics: _MemristorWeightStatics

    @staticmethod
    def init(
        device_pool: DevicePool,
        name: str,
        shape: abc.Sequence[int],
        weight_formula: Union[str, abc.abc.Callable[[jax.Array, jax.Array], jax.Array]],
        scale_factor: Union[np.float32, float],
        **_kwargs,
    ) -> MemristorWeight:
        pos, neg = device_pool.request_couple(shape, uid=name)

        return MemristorWeight.from_devices(
            (pos, neg), device_pool, name, weight_formula, scale_factor
        )
    
    def copy(self : MemristorWeight) -> MemristorWeight:
        return copy.deepcopy(self)

    @staticmethod
    def from_devices(
        devices: tuple[jax.Array, jax.Array],
        device_pool: DevicePool,
        name: str,
        weight_formula: Union[str, abc.abc.Callable[[jax.Array, jax.Array], jax.Array]],
        scale_factor: Union[np.float32, float],
    ):
        pos, neg = devices

        # if pos.shape != neg.shape:
        #     raise ValueError(
        #         f"Expected positive and negative devices of same shape, got {pos.shape = } v.s. {neg.shape = }"
        #     )

        shape = pos.shape

        return MemristorWeight(
            jnp.zeros(shape, dtype=np.uint16),
            jnp.zeros(shape, dtype=np.uint16),
            _MemristorWeightStatics(
                pos,
                neg,
                name,
                device_pool,
                weight_formula,
                scale_factor,
            ),
        )

    def replace_devices(self : MemristorWeight,
        devices: tuple[jax.Array, jax.Array]
    ) -> MemristorWeight:
        copy_obj = self.copy()
        pos, neg = jax.device_put(devices)
        return MemristorWeight.from_devices(
            (pos, neg), copy_obj.statics.pool, copy_obj.statics.name, copy_obj.statics.weight_formula, copy_obj.statics.scale_factor
        )

    @property
    def shape(self: MemristorWeight) -> tuple[int, ...]:
        return self.pos_pulse_id.shape

    @property
    def weight(self: MemristorWeight) -> jax.Array:
        wp = self.statics.pool.devices[self.statics.pos_dev_id, self.pos_pulse_id]
        wn = self.statics.pool.devices[self.statics.neg_dev_id, self.neg_pulse_id]

        if callable(self.statics.weight_formula):
            w = self.statics.weight_formula(wp, wn)
        elif self.statics.weight_formula == "diff":
            w = wp - wn
        elif self.statics.weight_formula == "conductance_diff":
            w = 1.0 / wn - 1.0 / wp
        elif self.statics.weight_formula == "log":
            w = jnp.log(wp / wn)
        else:
            raise NotImplementedError(
                f"Unknown weight formula '{self.statics.weight_formula}'"
            )

        return self.statics.scale_factor * w

    def incr_all(self: MemristorWeight, indices: abc.Iterable) -> MemristorWeight:
        indices = jnp.asarray(indices)
        assert indices.shape == self.pos_pulse_id.shape

        return MemristorWeight(
            self.pos_pulse_id + indices,
            self.neg_pulse_id,
            self.statics,
        )

    def decr_all(self: MemristorWeight, indices: abc.Iterable) -> MemristorWeight:
        indices = jnp.asarray(indices)
        assert indices.shape == self.neg_pulse_id.shape

        return MemristorWeight(
            self.pos_pulse_id,
            self.neg_pulse_id + indices,
            self.statics,
        )
    
    def reset(self: MemristorWeight, mask_pos : abc.Iterable = None, mask_neg : abc.Iterable = None) -> MemristorWeight:

        if mask_pos is None :
            mask_pos = jnp.ones_like(self.pos_pulse_id, dtype=bool)

        if mask_neg is None :
            mask_neg = jnp.ones_like(self.neg_pulse_id, dtype=bool)
        

        return MemristorWeight(
            jnp.where(mask_pos, 0, self.pos_pulse_id).astype(self.pos_pulse_id.dtype),
            jnp.where(mask_neg, 0, self.neg_pulse_id).astype(self.pos_pulse_id.dtype),
            
            # jnp.zeros(self.pos_pulse_id.shape, dtype = self.pos_pulse_id.dtype),
            # jnp.zeros(self.neg_pulse_id.shape, dtype = self.neg_pulse_id.dtype),
            self.statics,
        )
    
    def set_random(self: MemristorWeight) -> MemristorWeight:
        key_pos = jax.random.key(32) 
        key_neg = jax.random.split(key_pos, 1)[1]
        return MemristorWeight(
            jax.random.randint(key_pos, self.pos_pulse_id.shape, 0, 5000, dtype = self.pos_pulse_id.dtype),
            jax.random.randint(key_neg, self.neg_pulse_id.shape, 0, 5000, dtype = self.neg_pulse_id.dtype),
            self.statics,
        )
    
    def set_pulses(self: MemristorWeight, pos_pulses, neg_pulses) -> MemristorWeight:
        return MemristorWeight(
            pos_pulses,
            neg_pulses,
            self.statics,
        )



# We make it JAX-friendly
@jax.tree_util.register_static
class _MemristorWeightStatics(NamedTuple):
    pos_dev_id: jax.Array
    neg_dev_id: jax.Array

    name: str
    pool: DevicePool

    weight_formula: Union[str, abc.abc.Callable[[jax.Array, jax.Array], jax.Array]]
    scale_factor: Union[np.float32, float]






