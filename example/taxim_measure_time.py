#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
import torch

import time

from taxim.taxim_jax import TaximJax
from taxim.taxim_torch import TaximTorch


def get_hm(seed: int):
    # Generate some arbitrary heightmap
    height_map_init = np.zeros((480, 640), dtype=np.float32)
    rep = 100
    height_maps = np.zeros((2, rep, 480, 640), dtype=np.float32)
    rng = np.random.default_rng(seed)
    height_maps[..., 200:300, 200:500] = (
        rng.normal(0.0, 0.1, size=(2, rep, 100, 300)) + 10.0
    )
    return height_map_init, height_maps


def test_performance_jax(dev: str):
    dev = jax.devices(dev)[0]
    taxim = TaximJax(device=dev)
    # Generate some arbitrary heightmap
    height_map_init, height_maps = get_hm(0)
    height_maps = jax.device_put(height_maps, device=dev)
    height_map_init = jax.device_put(height_map_init, device=dev)

    # Warm up
    for i in range(10):
        taxim.render_direct(
            height_map_init, with_shadow=True, press_depth=1.0
        ).block_until_ready()
        taxim.render_direct(
            height_map_init, with_shadow=False, press_depth=1.0
        ).block_until_ready()

    hms = list(height_maps[0])
    start = time.time()
    for hm in hms:
        taxim.render_direct(hm, with_shadow=False, press_depth=1.0).block_until_ready()
    print(
        "Without shadow: {:8.4f}ms".format(
            1000 * (time.time() - start) / height_maps.shape[1]
        )
    )

    hms = list(height_maps[1])
    start = time.time()
    for hm in hms:
        taxim.render_direct(hm, with_shadow=True, press_depth=1.0).block_until_ready()
    print(
        "With shadow:    {:8.4f}ms".format(
            1000 * (time.time() - start) / height_maps.shape[1]
        )
    )


def test_performance_torch(dev: str):
    taxim = TaximTorch(device=dev)
    # Generate some arbitrary heightmap
    height_map_init, height_maps = get_hm(0)
    height_maps = torch.from_numpy(height_maps).to(dev)
    height_map_init = torch.from_numpy(height_map_init).to(dev)

    # Warm up
    for i in range(10):
        taxim.render_direct(height_map_init, with_shadow=True, press_depth=1.0)
        taxim.render_direct(height_map_init, with_shadow=False, press_depth=1.0)
    if dev == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for hm in height_maps[0]:
        taxim.render_direct(hm, with_shadow=False, press_depth=1.0)
        if dev == "cuda":
            torch.cuda.synchronize()
    print(
        "Without shadow: {:8.4f}ms".format(
            1000 * (time.time() - start) / height_maps.shape[1]
        )
    )

    start = time.time()
    for hm in height_maps[1]:
        taxim.render_direct(hm, with_shadow=True, press_depth=1.0)
        if dev == "cuda":
            torch.cuda.synchronize()
    print(
        "With shadow:    {:8.4f}ms".format(
            1000 * (time.time() - start) / height_maps.shape[1]
        )
    )


try:
    print("Accelerated implementation PyTorch (GPU)")
    test_performance_torch("cuda")
except:
    print("Accelerated implementation PyTorch (GPU) not available")

print()
print("Accelerated implementation PyTorch (CPU)")
test_performance_torch("cpu")

try:
    print()
    print("Accelerated implementation JAX (GPU)")
    test_performance_jax("cuda")
except:
    print("Accelerated implementation JAX (GPU) not available")

print()
print("Accelerated implementation JAX (CPU)")
test_performance_jax("cpu")

# Original implementation:
# Without shadow:  67.0822ms
# With shadow:    120.9922ms

# Accelerated implementation PyTorch (GPU)
# Without shadow:   3.2856ms
# With shadow:      8.2374ms

# Accelerated implementation PyTorch (CPU)
# Without shadow:  28.4607ms
# With shadow:    114.0626ms

# Accelerated implementation JAX (GPU)
# Without shadow:   0.7209ms
# With shadow:      1.6945ms

# Accelerated implementation JAX (CPU)
# Without shadow:  45.0090ms
# With shadow:    180.1877ms
