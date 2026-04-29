# Quantum Dynamics 3D Animator ⚛️🎥

A Python tool for computational chemists to visualize and animate electron density dynamics. 

This script parses directories of sequential `.molden` output files representing a molecular trajectory. It extracts geometries and molecular orbitals, calculates the electron density differences between two states (Attachment/Detachment densities), and renders a high-fidelity 3D `.mp4` animation.

## Features
* **Trajectory Alignment:** Uses the Kabsch algorithm to align rotating/translating molecules to a reference frame, preventing the 3D sampling grid from jittering during the animation.
* **High-Fidelity Rendering:** Powered by PyVista with Super-Sample Anti-Aliasing (SSAA), depth peeling (for accurate transparencies), and dynamic shadow mapping.
* **Interactive Camera Setup:** Includes an optional `--interactive` mode allowing you to manually rotate and zoom to the perfect camera angle before locking it in for the final render.
* **Numba-Accelerated:** Heavy atomic orbital (AO) evaluations on 3D grids are JIT-compiled for speed.

## Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies:

    pip install -r requirements.txt

*(Note: `imageio-ffmpeg` is required to encode the output frames into an H.264 `.mp4` video).*

## File Naming Convention

The script automatically detects the trajectory length and available electronic states based on your filenames. All `.molden` files in the target directory **must** strictly follow this format:
`[Name]_[FrameIndex]_[State].molden`

**Example:**
* `ethene_0_S0.molden` (Ground state, Frame 0)
* `ethene_0_S1.molden` (Excited state, Frame 0)
* `ethene_1_S0.molden` (Ground state, Frame 1)
* `ethene_1_S1.molden` (Excited state, Frame 1)

## Usage

Run the script from your terminal. If you don't provide a directory, it will look for `.molden` files in your current working directory.

    # Basic usage (runs in the current directory)
    python Atacchment_detacchment_v3.py

    # Specify a directory and customize the render
    python Atacchment_detacchment_v3.py /path/to/data --res 80 --iso 0.005 --fps 12

### Command Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `directory` | `str` | `cwd` | Target directory containing `.molden` files. |
| `--res` | `int` | `60` | Resolution of the 3D sampling grid. Higher = smoother blobs, slower render. |
| `--iso` | `float` | `0.002` | Cutoff value for the 3D density isosurfaces. |
| `--fps` | `int` | `8` | Frames per second for the output `.mp4` video. |
| `--opacity`| `float` | `0.5` | Transparency level of the density clouds (0.0 to 1.0). |
| `--view` | `str` | `iso` | Standard camera plane (`iso`, `xy`, `xz`, `yz`). |
| `--interactive`| `flag` | `False` | Opens an interactive 3D window to manually set the camera angle before rendering. |

## Dependencies 
* Make sure `molden_utils.py` is located in the same directory as the main script, as it handles the mathematical parsing and distance matrix computations.

## License
MIT License

## Acknowledgments
Work produced with the support of a 2024 Leonardo Grant for Scientific Research and Cultural Creation, BBVA Foundation.
