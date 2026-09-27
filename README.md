# BI_CE: Boltzmann Inversion & Cross-Entropy Coarse-Graining

A high-performance pipeline combining **Boltzmann Inversion (BI)** and **Cross-Entropy Method (CEM)** to map Full-Atomic (FA) polymer trajectories to Coarse-Grained (CG) bead-spring models using **OpenMM**.

Fitting targets include:
- Bond length distribution $P(l)$
- Bond angle distribution $P(\theta)$
- Radial distribution function $g(r)$ (RDF)

---

## Installation

Ensure you have Python 3.9+ installed. Install the dependencies via:

```bash
pip install -r requirements.txt
```

### Dependencies:
- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `MDAnalysis>=2.0.0`
- `openmm>=8.0.0`

---

## CLI Usage

You can run the pipeline directly via `main.py`:

```bash
./main.py -s <structure_file> -t <trajectory_file> [options]
```

### Basic Example (Mapping & Verification MD):
```bash
./main.py -s ./FA/pe_l.psf -t ./FA/1nptts1.dcd --steps 100
```

### With CEM Optimization:
```bash
./main.py -s ./FA/pe_l.psf -t ./FA/1nptts1.dcd \
    --steps 100 \
    --cem \
    --samples 20 \
    --processes 4 \
    --cem-md-steps 50
```

### On a Specific GPU:
```bash
./main.py -s ./FA/pe_l.psf -t ./FA/1nptts1.dcd --gpu-id 0 --platform CUDA
```
> **Note**: Multiple subprocesses during CEM optimization will automatically share and run concurrently on the specified GPU device (`gpu-id`). If no GPU is available, it automatically falls back to multi-process parallel execution on CPU.

---

## Command-Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `-s, --structure` | Path to topology/structure file (`.psf`, `.pdb`, `.gro`, etc.) | **Required** |
| `-t, --trajectory` | Path to trajectory file (`.dcd`, `.xtc`, `.trr`, etc.) | **Required** |
| `-o, --output` | Output prefix for generated CG coordinate/data files | `./CG` |
| `--temp, --temperature` | Target simulation temperature in Kelvin | `300.0` |
| `--press, --pressure` | Target pressure in atm | `1.0` |
| `--mass` | Bead mass in amu (auto-calculated if omitted) | Auto |
| `--steps` | Number of verification MD steps in OpenMM | `100` |
| `--dt` | MD timestep in femtoseconds | `4.0` |
| `--gpu-id` | Target GPU device index | `0` |
| `--platform` | OpenMM platform (`CUDA`, `OpenCL`, `CPU`, `Reference`) | Auto |
| `--cem` | Flag to trigger CEM optimization after Boltzmann Inversion | `False` |
| `--samples` | Number of parameter candidates sampled per CEM epoch | `20` |
| `--processes` | Number of parallel worker processes for CEM | `4` |
| `--cem-md-steps` | Simulation steps per sample during CEM evaluation | `50` |
| `--export-data` | Export CG LAMMPS data file (`.data`) | `False` |
