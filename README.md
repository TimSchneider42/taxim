# GPU Accelerated Implementation of the Taxim Tactile Simulator

GPU accelerated implementation of the optical simulation of the [Taxim Simulator](https://ieeexplore.ieee.org/abstract/document/9681378?casa_token=BV_S7sgid7YAAAAA:cwQLbde1HRXkKxu15s1g7QWkTSvW9UP4ziocxWS50mS44aumyEveuQh9VGnPJzzBmE7jRkXC).
Both PyTorch and JAX backends are available.
Note that this project is not the official implementation by the authors, which can be found [here](https://github.com/CMURoboTouch/Taxim).
Because of efficient use of vectorization and GPU acceleration, this implementation achieves significant speed-ups for image generation with and without shadows compared to the official implementation.

Here is a comparison of the render times for a single frame for the different implementations:

| Implementation | Without shadow | With shadow |
|----------------|----------------|-------------|
| Original       | 67.08ms        | 120.99ms    |
| Torch CPU      | 28.46ms        | 114.06ms    |
| Torch GPU      | 3.29ms         | 8.24ms      |
| JAX CPU        | 45.01ms        | 180.19ms    |
| JAX GPU        | **0.72ms**     | **1.69ms**  |

The speed-up is around **x21** for image generation without shadow and around **x25** with shadow.

## Installation

Taxim can be installed via
```bash
pip install taxim[OPTIONS]
```
where OPTIONS can be any subset of the following:
- `torch`: Install the PyTorch backend.
- `jax`: Install the JAX backend with CUDA 12 support.
- `jax-cpu`: Install the JAX backend without GPU support.
- `examples`: Install dependencies for running the examples.
- `all`: Install all of the above.

Note that either `torch`, `jax`, or `jax-cpu` has to be chosen as a backend for Taxim to work.

Depending on the installed CUDA version, _PyTorch_, _torch-scatter_, and _JAX_ might have to be installed manually.
Follow the instructions on their respective websites:
- [PyTorch](https://pytorch.org/get-started/locally/)
- [torch-scatter](https://pypi.org/project/torch-scatter/)
- [JAX](https://jax.readthedocs.io/en/latest/installation.html)

## Usage

To use the tactile simulator, first create an instance of the `Taxim` class:

```python
import torch
from taxim import Taxim

dev = "cuda"

taxim = Taxim(device=dev)
```

Optionally, you can pass the constructor a path to a calibration folder if you wish to use a different calibration.

Now you can use this instance to render a height map into a tactile image:

```python
# Generate some arbitrary heightmap
height_map = torch.zeros((taxim.height, taxim.width), device=dev)
height_map[200:300, 200:500] = -10.0

# Render an image using this height map
img = taxim.render(height_map, with_shadow=True, press_depth=1.0)
```

The values of the height map correspond to the distance of the object to the highest (furthest from the camera) point of the gel in mm.
Hence, a value of 0 means that the corresponding point is at the same height as the highest point of the gel.
The smaller the values of the height map are, the closer are the corresponding points to the camera.
For a point to be in contact with the gel, it has to have a value smaller or equal to zero.

Note that this definition of the height map differs from the original definition by the authors, who assume that higher values mean that points are closer, and use a different reference point.
If you want to recover the original behavior of Taxim, just pass `orig_hm_fmt=True` when calling `taxim.render`:
```python
# Render an image using this height map
img = taxim.render(height_map, with_shadow=True, press_depth=1.0, orig_hm_fmt=True)
```

For a usage example refer to `example/taxim_visualize.py`.

### Using a different calibration

To use a different calibration, you must create a folder with the following files and pass its path as an argument to
the `Taxim` constructor:

1. `dataPack.npz`: contains the recorded data. Actually only the first frame is needed for generating the background,
   but we keep it like it is to be compatible to the original Taxim calibration files.
2. `gelmap.npy`: contains the height map of the gel pad.
3. `params.json`: contains parameters for the simulator. For an example, refer to `taxim/calib/params.json`
4. `polycalib.npz`: contains the polynomial coefficients for the gel simulation.
5. `shadowTable.npz`: contains the shadow table.

For further explanation on how to obtain these files, please refer to
the [original implementation](https://github.com/CMURoboTouch/Taxim).
