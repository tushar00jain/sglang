# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/utils.py

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import dataclasses
import pickle
import time
from collections import deque
from collections.abc import Sequence
from typing import Any

import torch
from torch.distributed import TCPStore

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Backend routing for the ``split_group`` subgroup path.
#
# sglang-diffusion owns its own world process group (see
# ``parallel_state.init_distributed_environment``) and pushes it into srt's
# globals, so the routing below decides the backend for *both* frameworks in a
# diffusion-initiated process.
# ---------------------------------------------------------------------------

# split_group selects a subset of the parent PG's *per-device* backends, so the
# world PG must be device-qualified and it must carry "cpu:gloo" for the CPU
# subgroups to have a gloo backend to filter for.
SPLIT_GROUP_WORLD_BACKEND = "cpu:gloo,cuda:nccl"

# Filter for a device (CUDA collective) subgroup split off the world PG.
SPLIT_GROUP_DEVICE_BACKEND = "cuda:nccl"

# Filter for a CPU-coordination subgroup split off the world PG. It has to keep
# "cuda:nccl" as well: ProcessGroup::splitGroup requires the deviceTypes filter
# to include the parent's default backend device type (cuda here), so a pure
# "cpu:gloo" split is rejected outright. The resulting group is compound, which
# is fine for CPU collectives/P2P and for dist.monitored_barrier (it checks for
# a CPU-capable backend via group._device_types, not for the literal "gloo"
# name).
SPLIT_GROUP_CPU_BACKEND = "cpu:gloo,cuda:nccl"

# Backend for genuinely per-peer P2P groups (pipeline parallel). "nccl-lazy"
# defers communicator creation until first use so send/recv to different stages
# can overlap. These groups are built members-only (see below).
P2P_DEVICE_BACKEND = "nccl-lazy"


def route_world_backend(backend: str | None) -> str | None:
    """Device-qualify a requested world backend for the split_group path.

    Mirrors ``sglang.srt.distributed.parallel_state.init_distributed_environment``:
    a bare ``nccl`` becomes ``cpu:gloo,cuda:nccl``. Non-CUDA backends
    (gloo/hccl/...) are passed through untouched.
    """
    if backend in ("nccl", "cuda:nccl"):
        return SPLIT_GROUP_WORLD_BACKEND
    return backend


def can_split_world() -> bool:
    """Whether the world PG can have subgroups carved off it with split_group.

    Derived from the *live* world PG rather than from the requested backend
    string: the world may have been initialized by someone else (srt, a test
    harness, or an embedding trainer) in a shape split_group cannot use. Those
    worlds must keep the upstream ``new_group`` path.
    """
    if not torch.distributed.is_initialized():
        return False
    world_pg = torch.distributed.group.WORLD
    if world_pg is None:
        return False
    # split_group requires an eagerly device-bound parent communicator.
    if getattr(world_pg, "bound_device_id", None) is None:
        return False
    # ... and a device-qualified parent carrying a cpu backend, so that both the
    # device and the CPU subgroup filters resolve against it.
    return "cpu:" in torch.distributed.get_backend(world_pg)


def ensure_divisibility(numerator, denominator) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tuple(tensor_list)


@dataclasses.dataclass
class StatelessProcessGroup:
    """A dataclass to hold a metadata store, and the rank, world_size of the
    group. Only use it to communicate metadata between processes.
    For data-plane communication, create NCCL-related objects.
    """

    rank: int
    world_size: int
    store: torch._C._distributed_c10d.Store
    data_expiration_seconds: int = 3600  # 1 hour

    # dst rank -> counter
    send_dst_counter: dict[int, int] = dataclasses.field(default_factory=dict)
    # src rank -> counter
    recv_src_counter: dict[int, int] = dataclasses.field(default_factory=dict)
    broadcast_send_counter: int = 0
    broadcast_recv_src_counter: dict[int, int] = dataclasses.field(default_factory=dict)

    # A deque to store the data entries, with key and timestamp.
    entries: deque[tuple[str, float]] = dataclasses.field(default_factory=deque)

    def __post_init__(self):
        assert self.rank < self.world_size
        self.send_dst_counter = {i: 0 for i in range(self.world_size)}
        self.recv_src_counter = {i: 0 for i in range(self.world_size)}
        self.broadcast_recv_src_counter = {i: 0 for i in range(self.world_size)}

    def send_obj(self, obj: Any, dst: int):
        """Send an object to a destination rank."""
        self.expire_data()
        key = f"send_to/{dst}/{self.send_dst_counter[dst]}"
        self.store.set(key, pickle.dumps(obj))
        self.send_dst_counter[dst] += 1
        self.entries.append((key, time.perf_counter()))

    def expire_data(self) -> None:
        """Expire data that is older than `data_expiration_seconds` seconds."""
        while self.entries:
            # check the oldest entry
            key, timestamp = self.entries[0]
            if time.perf_counter() - timestamp > self.data_expiration_seconds:
                self.store.delete_key(key)
                self.entries.popleft()
            else:
                break

    def recv_obj(self, src: int) -> Any:
        """Receive an object from a source rank."""
        obj = pickle.loads(
            self.store.get(f"send_to/{self.rank}/{self.recv_src_counter[src]}")
        )
        self.recv_src_counter[src] += 1
        return obj

    def broadcast_obj(self, obj: Any | None, src: int) -> Any:
        """Broadcast an object from a source rank to all other ranks.
        It does not clean up after all ranks have received the object.
        Use it for limited times, e.g., for initialization.
        """
        if self.rank == src:
            self.expire_data()
            key = f"broadcast_from/{src}/" f"{self.broadcast_send_counter}"
            self.store.set(key, pickle.dumps(obj))
            self.broadcast_send_counter += 1
            self.entries.append((key, time.perf_counter()))
            return obj
        else:
            key = f"broadcast_from/{src}/" f"{self.broadcast_recv_src_counter[src]}"
            recv_obj = pickle.loads(self.store.get(key))
            self.broadcast_recv_src_counter[src] += 1
            return recv_obj

    def all_gather_obj(self, obj: Any) -> list[Any]:
        """All gather an object from all ranks."""
        gathered_objs = []
        for i in range(self.world_size):
            if i == self.rank:
                gathered_objs.append(obj)
                self.broadcast_obj(obj, src=self.rank)
            else:
                recv_obj = self.broadcast_obj(None, src=i)
                gathered_objs.append(recv_obj)
        return gathered_objs

    def barrier(self):
        """A barrier to synchronize all ranks."""
        for i in range(self.world_size):
            if i == self.rank:
                self.broadcast_obj(None, src=self.rank)
            else:
                self.broadcast_obj(None, src=i)

    @staticmethod
    def create(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        data_expiration_seconds: int = 3600,
    ) -> "StatelessProcessGroup":
        """A replacement for `torch.distributed.init_process_group` that does not
        pollute the global state.

        If we have process A and process B called `torch.distributed.init_process_group`
        to form a group, and then we want to form another group with process A, B, C,
        D, it is not possible in PyTorch, because process A and process B have already
        formed a group, and process C and process D cannot join that group. This
        function is a workaround for this issue.

        `torch.distributed.init_process_group` is a global call, while this function
        is a stateless call. It will return a `StatelessProcessGroup` object that can be
        used for exchanging metadata. With this function, process A and process B
        can call `StatelessProcessGroup.create` to form a group, and then process A, B,
        C, and D can call `StatelessProcessGroup.create` to form another group.
        """  # noqa
        store = TCPStore(
            host_name=host,
            port=port,
            world_size=world_size,
            is_master=(rank == 0),
        )

        return StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            data_expiration_seconds=data_expiration_seconds,
        )
