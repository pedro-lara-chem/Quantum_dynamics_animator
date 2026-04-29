"""
Molden Parsing and Quantum Chemistry Evaluation Utilities

This module provides high-performance utilities for parsing computational chemistry 
output files (.molden) and evaluating Gaussian Type Orbitals (GTOs) on 3D grids. 
It leverages Scipy's optimized spatial distance functions and Numba's Just-In-Time 
(JIT) compilation to massively accelerate radial and angular mathematical operations 
for large molecular trajectories.

Author: Pedro Lara
License: MIT
"""

import numpy as np
import numba
from scipy.spatial.distance import pdist, squareform
from typing import List, Dict, Tuple, Any

# --- Constants ---
# Maps string characters from molden files to angular momentum integers
L_QUANTUM_NUMBERS_MAP: Dict[str, int] = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4}

# Standard order of solid harmonic angular functions (PySCF convention)
ANGULAR_LABELS: Dict[int, List[str]] = {
    0: ['s'],
    1: ['px', 'py', 'pz'],
    2: ['dxy', 'dyz', 'dz2', 'dxz', 'dx2y2'],
    3: ['fz3', 'fxz2', 'fyz2', 'fz(x2-y2)', 'fxyz', 'fx(x2-3y2)', 'fy(3x2-y2)']
}

# --- Distance Matrix ---
def distance_matrix(coordinates: np.ndarray, n_atoms: int, threshold: float = 3.0) -> np.ndarray:
    """
    Calculates a symmetric distance matrix for all atoms in a molecule.
    Uses Scipy's highly optimized C-backend instead of standard nested loops.

    Args:
        coordinates (np.ndarray): Nx3 array of atomic Cartesian coordinates.
        n_atoms (int): The number of atoms.
        threshold (float): Maximum bond length in Bohr. Distances above this are set to 0.

    Returns:
        np.ndarray: An NxN symmetric distance matrix.
    """
    if n_atoms <= 1:
        return np.zeros((n_atoms, n_atoms))
    
    # pdist computes pairwise distances; squareform turns it into a symmetric NxN matrix
    dist_mat = squareform(pdist(coordinates, metric='euclidean'))
    
    # Vectorized thresholding: ignore bonds larger than the threshold (e.g., non-bonded atoms)
    dist_mat[dist_mat > threshold] = 0.0
    return dist_mat

# --- Data Parsing ---
def parse_molden_file(filepath: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parses a Molden file to extract atomic coordinates, Gaussian Basis Sets (GTOs), 
    and Molecular Orbital (MO) coefficients.

    Args:
        filepath (str): Absolute or relative path to the .molden file.

    Returns:
        Tuple containing three lists:
            - atoms_data: Dictionaries of atomic properties (label, coords, atomic_number).
            - gto_data: Dictionaries mapping atoms to their Gaussian shells and primitives.
            - mo_data: Dictionaries of MO properties (energy, occupancy, coefficients).
    """
    atoms_data: List[Dict[str, Any]] = []
    gto_data: List[Dict[str, Any]] = []
    mo_data: List[Dict[str, Any]] = []

    current_section: str = ""
    current_atom_gto_shells: List[Dict[str, Any]] = []
    current_atom_gto_idx: int = -1
    current_mo_coeffs: List[float] = []
    current_mo_details: Dict[str, Any] = {}
    atom_units: str = "AU"

    try:
        with open(filepath, 'r') as f_iter:
            for line_number, raw_line in enumerate(f_iter, 1):
                line = raw_line.strip()
                if not line: continue

                # Detect section headers (e.g., [Atoms], [GTO], [MO])
                if line.startswith('['):
                    # Finalize pending MO data before switching sections
                    if current_section == "MO" and current_mo_details:
                        if current_mo_coeffs: 
                            current_mo_details['coefficients'] = list(current_mo_coeffs)
                            mo_data.append(current_mo_details)
                        current_mo_coeffs, current_mo_details = [], {}
                    
                    if line.lower().startswith("[atoms]"):
                        # Finalize pending GTO data
                        if current_section == "GTO" and current_atom_gto_shells and current_atom_gto_idx != -1:
                            gto_data.append({'atom_index': current_atom_gto_idx, 'shells': list(current_atom_gto_shells)})
                            current_atom_gto_shells, current_atom_gto_idx = [], -1
                        current_section = "Atoms"
                        atom_units = "Angs" if "angs" in line.lower() else "AU"
                        
                    elif line.lower().startswith("[gto]"):
                        current_section, current_atom_gto_idx, current_atom_gto_shells = "GTO", -1, []
                        
                    elif line.lower().startswith("[mo]"):
                        if current_section == "GTO" and current_atom_gto_shells and current_atom_gto_idx != -1:
                            gto_data.append({'atom_index': current_atom_gto_idx, 'shells': list(current_atom_gto_shells)})
                            current_atom_gto_shells, current_atom_gto_idx = [], -1 
                        current_section = "MO"
                    else:
                        if current_section == "GTO" and current_atom_gto_shells and current_atom_gto_idx != -1:
                            gto_data.append({'atom_index': current_atom_gto_idx, 'shells': list(current_atom_gto_shells)})
                            current_atom_gto_shells, current_atom_gto_idx = [], -1
                        current_section = line 
                    continue

                # Parse geometry
                if current_section == "Atoms":
                    parts = line.split()
                    if len(parts) == 6:
                        atoms_data.append({
                            'label': parts[0], 'number_in_molden': int(parts[1]),
                            'atomic_number': int(parts[2]), 'x': float(parts[3]),
                            'y': float(parts[4]), 'z': float(parts[5]), 'unit': atom_units
                        })

                # Parse basis sets
                elif current_section == "GTO":
                    parts = line.split()
                    # Check if line indicates a new atom index
                    if len(parts) <= 2 and all(p.isdigit() for p in parts) and parts:
                        if current_atom_gto_shells and current_atom_gto_idx != -1 :
                            gto_data.append({'atom_index': current_atom_gto_idx, 'shells': list(current_atom_gto_shells)})
                        
                        atom_seq_num_gto = int(parts[0]) 
                        found_atom_idx = next((i for i, a in enumerate(atoms_data) if a['number_in_molden'] == atom_seq_num_gto), -1)
                        current_atom_gto_idx = found_atom_idx if found_atom_idx != -1 else atom_seq_num_gto - 1 
                        current_atom_gto_shells = []
                        
                    # Parse specific shell configuration (s, p, d, etc.)
                    elif parts and parts[0].isalpha() and parts[0].lower() in ['s', 'p', 'sp', 'd', 'f', 'g', 'h', 'i']:
                        shell_type, num_primitives, scale_factor = parts[0].lower(), 0, 1.0
                        if len(parts) == 3:
                            num_primitives, scale_factor = int(parts[1]), float(parts[2])
                        elif len(parts) == 2:
                            num_primitives = int(parts[1])
                        
                        current_shell_primitives = []
                        for _ in range(num_primitives):
                            prim_parts = next(f_iter).strip().split()
                            current_shell_primitives.append({
                                'exponent': float(prim_parts[0]),
                                'coefficients': [float(c) for c in prim_parts[1:]]
                            })
                        current_atom_gto_shells.append({
                            'type': shell_type, 'scale_factor': scale_factor, 'primitives': current_shell_primitives
                        })

                # Parse orbital parameters
                elif current_section == "MO":
                    if line.lower().startswith("sym="): 
                        if current_mo_details and current_mo_coeffs: 
                            current_mo_details['coefficients'] = list(current_mo_coeffs)
                            mo_data.append(current_mo_details)
                        current_mo_details, current_mo_coeffs = {'symmetry': line.split('=')[1].strip()}, []
                    elif line.lower().startswith("ene="):
                        current_mo_details['energy'] = float(line.split('=')[1].strip())
                    elif line.lower().startswith("spin="):
                        current_mo_details['spin'] = line.split('=')[1].strip()
                    elif line.lower().startswith("occup="):
                        current_mo_details['occupancy'] = float(line.split('=')[1].strip())
                    elif len(line.split()) == 2: 
                        current_mo_coeffs.append((float(line.split()[1])))

            # Final cleanup check for EOF
            if current_section == "GTO" and current_atom_gto_shells and current_atom_gto_idx != -1:
                gto_data.append({'atom_index': current_atom_gto_idx, 'shells': list(current_atom_gto_shells)})
            if current_section == "MO" and current_mo_details and current_mo_coeffs: 
                current_mo_details['coefficients'] = list(current_mo_coeffs)
                mo_data.append(current_mo_details)

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return [], [], []
    return atoms_data, gto_data, mo_data

# --- Accelerated Math Functions (Numba) ---

@numba.njit(cache=True)
def numba_factorial2(n_val: float) -> float:
    """Computes the double factorial n!!, optimized with Numba."""
    if n_val < -1: return 0.0
    if n_val in (-1, 0): return 1.0
    res, val = 1.0, float(n_val)
    while val > 0.5:
        res *= val
        val -= 2.0
    return res

@numba.njit(cache=True)
def norm_primitive_numba(alpha: float, l_val: int) -> float:
    """Computes the normalization constant for a primitive Gaussian Type Orbital."""
    if alpha < 1e-12: return 0.0
    double_fact_val = numba_factorial2(2 * l_val - 1)
    return ((2.0 * alpha / np.pi)**0.75 * (2.0 * np.sqrt(alpha))**l_val) / np.sqrt(double_fact_val)

@numba.njit(cache=True)
def compute_radial_part_numba(r_sq_grid_points: np.ndarray, exponents_arr: np.ndarray, 
                              coeffs_arr: np.ndarray, scale_factor_sq: float, l_val: int) -> np.ndarray:
    """Sums the contracted radial components of a Gaussian primitive."""
    sum_contracted_radial_part = np.zeros(r_sq_grid_points.shape[0], dtype=np.float64)
    for i in range(exponents_arr.shape[0]):
        if abs(coeffs_arr[i]) < 1e-15: continue
        alpha_scaled = exponents_arr[i] * scale_factor_sq
        N_k_prim = norm_primitive_numba(alpha_scaled, l_val)
        if abs(N_k_prim) < 1e-15: continue
        sum_contracted_radial_part += coeffs_arr[i] * N_k_prim * np.exp(-alpha_scaled * r_sq_grid_points)
    return sum_contracted_radial_part

@numba.njit(cache=True)
def real_sph_harmonics_pyscf_order_numba(l: int, theta: np.ndarray, phi: np.ndarray) -> List[np.ndarray]:
    """Generates the solid harmonic angular components up to f-orbitals (l=3)."""
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    
    if l == 0: 
        return [np.ones_like(theta)]
    elif l == 1:
        return [sin_t * cos_p, sin_t * sin_p, cos_t]
    elif l == 2:
        return [sin_t**2 * cos_p * sin_p, sin_t * cos_t * sin_p, (3.0 * cos_t**2 - 1.0) / 2.0, 
                sin_t * cos_t * cos_p, sin_t**2 * (cos_p**2 - sin_p**2)]
    elif l == 3:
        return [cos_t * (5.0 * cos_t**2 - 3.0) / 2.0, sin_t * cos_p * (5.0 * cos_t**2 - 1.0),
                sin_t * sin_p * (5.0 * cos_t**2 - 1.0), cos_t * sin_t**2 * np.cos(2*phi),
                sin_t**2 * cos_t * sin_p * cos_p, sin_t**3 * np.cos(3*phi), sin_t**3 * np.sin(3*phi)]
    return [np.empty(0, dtype=np.float64)]

def compute_atomic_orbitals(grid_points: np.ndarray, atoms_data: List[Dict[str, Any]], 
                            gto_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """
    Evaluates the full set of Atomic Orbitals on a defined Cartesian 3D grid.

    Args:
        grid_points (np.ndarray): Px3 array of coordinates (P = number of points).
        atoms_data (List[Dict]): Parsed atomic properties.
        gto_data (List[Dict]): Parsed basis set data.

    Returns:
        Tuple:
            - np.ndarray: PxA matrix where P is grid points and A is total atomic orbitals.
            - List[str]: Optional list of labels (returned empty here for speed).
    """
    num_grid_points = grid_points.shape[0]
    total_num_aos = 0

    # Pre-calculate the exact shape of our matrix to avoid dynamic resizing in memory
    for atom_gto_idx, atom_gto_info in enumerate(gto_data):
        for shell_info in atom_gto_info['shells']:
            shell_type_full = shell_info['type'].lower()
            for current_coeff_idx, char_l in enumerate(shell_type_full):
                if char_l not in L_QUANTUM_NUMBERS_MAP or L_QUANTUM_NUMBERS_MAP[char_l] not in ANGULAR_LABELS: continue
                if shell_info['primitives'] and current_coeff_idx < len(shell_info['primitives'][0]['coefficients']):
                    angular_momentum_label_suffixes = ANGULAR_LABELS[L_QUANTUM_NUMBERS_MAP[char_l]]
                    total_num_aos += len(angular_momentum_label_suffixes)

    if total_num_aos == 0: 
        return np.empty((num_grid_points, 0), dtype=np.float64), []

    # Pre-allocate output array
    ao_matrix = np.empty((num_grid_points, total_num_aos), dtype=np.float64)
    current_ao_matrix_idx = 0

    # Build the AOs
    for atom_gto_info in gto_data:
        atom_idx = atom_gto_info['atom_index']
        atom_center = np.array([atoms_data[atom_idx][k] for k in ('x', 'y', 'z')], dtype=np.float64)
        
        # Shift points relative to the atomic center
        R_vectors = grid_points - atom_center  
        x_rel, y_rel, z_rel = R_vectors[:, 0], R_vectors[:, 1], R_vectors[:, 2]

        # Convert to spherical coordinates
        r_sq = x_rel**2 + y_rel**2 + z_rel**2
        r_stable = np.sqrt(r_sq) 
        theta_vals = np.arccos(np.clip(z_rel / (r_stable + 1e-12), -1.0, 1.0))
        phi_vals = np.arctan2(y_rel, x_rel) 
        
        for shell_info in atom_gto_info['shells']:
            for current_coeff_idx, char_l in enumerate(shell_info['type'].lower()):
                if char_l not in L_QUANTUM_NUMBERS_MAP: continue
                l_val = L_QUANTUM_NUMBERS_MAP[char_l]
                
                exponents_list, coeffs_list = [], []
                for prim_data in shell_info['primitives']:
                    if current_coeff_idx < len(prim_data['coefficients']):
                        exponents_list.append(prim_data['exponent'])
                        coeffs_list.append(prim_data['coefficients'][current_coeff_idx])

                if not exponents_list: continue

                # Evaluate Math Parts
                sum_gaussians_part = compute_radial_part_numba(
                    r_sq, np.array(exponents_list), np.array(coeffs_list), shell_info['scale_factor']**2, l_val
                )
                Slm_angular_parts_list = real_sph_harmonics_pyscf_order_numba(l_val, theta_vals, phi_vals)
                r_pow_l = np.power(r_stable, l_val)

                # Assemble Full Wavefunction Component
                for ang_part in Slm_angular_parts_list:
                    if current_ao_matrix_idx < total_num_aos:
                        ao_matrix[:, current_ao_matrix_idx] = sum_gaussians_part * r_pow_l * ang_part
                        current_ao_matrix_idx += 1

    return ao_matrix[:, :current_ao_matrix_idx], []

def compute_density_matrix(mo_coeffs: List[List[float]], mo_occupancies: List[float]) -> np.ndarray:
    """
    Computes the Atomic Orbital (AO) basis density matrix.

    Args:
        mo_coeffs (List[List[float]]): Molecular Orbital coefficients.
        mo_occupancies (List[float]): Occupancy values for each MO.

    Returns:
        np.ndarray: The density matrix.
    """
    n_aos = len(mo_coeffs[0])
    D = np.zeros((n_aos, n_aos))
    for i, occ in enumerate(mo_occupancies):
        if occ > 1e-5:
            c_vec = np.array(mo_coeffs[i])
            # D_uv = Sum_i (occ_i * c_ui * c_vi) -> Outer product
            D += occ * np.outer(c_vec, c_vec)
    return D