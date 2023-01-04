"""
Utilities for managing gpus.
References:
https://github.com/google/ldif/tree/master/ldif/util
"""

import sys

import subprocess as sp

from impax.utils.file_util import log


def get_free_gpu_memory(cuda_device_index):
    """Returns the current # of free megabytes for the specified device."""
    if sys.platform == "darwin":
        # No GPUs on darwin...
        return 0
    result = sp.check_output(
        "nvidia-smi --query-gpu=memory.free " "--format=csv,nounits,noheader",
        shell=True,
    )
    result = result.decode("utf-8").split("\n")[:-1]
    log.verbose(f"The system has {len(result)} gpu(s).")
    free_mem = int(result[cuda_device_index])
    log.info(f"The {cuda_device_index}-th GPU has {free_mem} MB free.")
    if cuda_device_index >= len(result):
        raise ValueError(f"Couldn't parse result for GPU #{cuda_device_index}")
    return int(result[cuda_device_index])


def get_allowable_fraction_without(mem_to_reserve, cuda_device_index):
    """Returns the fraction to give to tensorflow after reserving x megabytes."""
    current_free = get_free_gpu_memory(cuda_device_index)
    allowable = current_free - mem_to_reserve  # 1GB
    allowable_fraction = allowable / current_free
    if allowable_fraction <= 0.0:
        raise ValueError(
            f"Can't leave 1GB over for the inference kernel, because"
            f" there is only {allowable} total free GPU memory."
        )
    return allowable_fraction
