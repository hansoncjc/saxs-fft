# saxs-fft

Structure factor and SAS intensity calculation from GSD files using FFT.

## Installation

Install the package using pip:

```bash
# Development mode (editable install)
pip install -e .

# With dev dependencies
pip install -e ".[dev]"

# from github
pip install git+https://github.com/hansoncjc/saxs-fft.git
```


## Quick Start

### Basic Usage
```python
from saxs_fft import StructureFactor
import torch

# Create structure factor calculator
sf = StructureFactor(
    gsd_path,
    N_grid,
    frames="last:10",
    device=torch.device("cuda"),  # or "cpu"
    dtype=torch.float64
)

# Compute 1D structure factor
q, s = sf.compute_s_1d()
```

### Example Notebooks
- `examples/Sphere.ipynb` - Basic sphere example
- `examples/Sphere_pytorch.ipynb` - PyTorch/GPU example

## Features

### Frame Selection
`StructureFactor(..., frames=...)` accepts:
- `'all'` - Process all frames
- `int` - Single frame index
- iterable of `int` - Specific frame indices
- `'last:N'` - Last N frames (default: `'last:5'`)

### GPU Support
The package includes torch-backed structure factor calculations with GPU support:

```python
from saxs_fft import StructureFactor
import torch

# Use CUDA GPU
sf = StructureFactor(
    gsd_path,
    N_grid,
    frames="last:10",
    device=torch.device("cuda"),
    dtype=torch.float64
)

# Or use CPU
sf = StructureFactor(
    gsd_path,
    N_grid,
    frames="last:10",
    device=torch.device("cpu"),
    dtype=torch.float64
)
```

## Testing

Run unit tests:
```bash
python tests/single_sim_test.py
python tests/structurefactor_test.py
```

GPU benchmark:
```bash
python tests/cpu_gpu_bench.py
```

## API Reference

Main classes and functions:
- `StructureFactor` - Calculate structure factors from GSD files
- `Sphere` - Form factor for spherical particles
- `SASIntensity` - Calculate SAS intensity
- `extract_positions` - Extract particle positions from GSD files
- `read_configuration` - Read configuration data

## Requirements

- Python >=3.8
- numpy
- torch
- gsd
- matplotlib
