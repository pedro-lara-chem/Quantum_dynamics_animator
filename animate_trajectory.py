"""
Quantum Dynamics 3D Animator

This script parses directories of sequential computational chemistry output files, 
extracts transition state geometries and orbitals, evaluates electron density 
differences (Attachment/Detachment densities), and renders high-fidelity 3D 
trajectory animations using PyVista and FFMPEG.

Author: Pedro Lara 
License: MIT
"""

import numpy as np
import pyvista as pv
import os
import argparse
import re
import glob
import shutil
import imageio.v2 as imageio
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from typing import List, Tuple, Optional
from tqdm import tqdm  # Added for progress bar

# Dependency injection from custom utilities module
from molden_utils import (
    distance_matrix, parse_molden_file, 
    compute_atomic_orbitals, compute_density_matrix
)

def detect_trajectory_info(target_path: str) -> Tuple[str, List[int], List[str]]:
    files = [f for f in os.listdir(target_path) if f.endswith('.molden')]
    if not files: 
        raise FileNotFoundError(f"No .molden files found in {target_path}")

    pattern = re.compile(r"([a-zA-Z0-9]+)_(\d+)_([a-zA-Z]\d+)\.molden")
    matches = [m for m in (pattern.match(f) for f in files) if m]
    if not matches: 
        raise ValueError("Files must follow 'name_frame_state.molden' format.")

    base_name = matches[0].group(1)
    frames = sorted(list(set(int(m.group(2)) for m in matches)))
    states = sorted(list(set(m.group(3) for m in matches)))
    
    return base_name, frames, states

def get_alignment_transform(reference_coords: np.ndarray, target_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu_ref = np.mean(reference_coords, axis=0)
    mu_tgt = np.mean(target_coords, axis=0)
    
    ref_centered = reference_coords - mu_ref
    tgt_centered = target_coords - mu_tgt
    
    cov_matrix = np.dot(tgt_centered.T, ref_centered)
    v, _, w_t = np.linalg.svd(cov_matrix)
    
    R = np.dot(v, w_t)
    
    if np.linalg.det(R) < 0:
        v[:, -1] *= -1
        R = np.dot(v, w_t)
        
    return R, mu_ref, mu_tgt

def plot3D(coordinates: np.ndarray, n_atoms: int, symbols: List[str], 
           dist_mat: np.ndarray, plotter: pv.Plotter) -> None:
    color_dict_rgb = {
        'C': 'black', 'H': 'gainsboro', 'O': 'red', 'F': 'cyan',
        'Cl': 'green', 'N': 'blue', 'S': 'yellow', 'Br': 'darkred',
        'B': 'pink', 'Al': 'brown', 'Fe': 'orange'
    }
    radius_data = {
        'C': 0.70, 'H': 0.31, 'O': 0.48, 'F': 0.42, 'Cl': 0.79,
        'N': 0.56, 'S': 1.00, 'Br': 1.14, 'B': 0.85, 'Al': 1.43, 'Fe': 1.26
    }
    
    for i in range(n_atoms):
        col = color_dict_rgb.get(symbols[i], 'gray')
        rad = radius_data.get(symbols[i], 0.5)
        sphere = pv.Sphere(radius=rad, center=coordinates[i])
        plotter.add_mesh(sphere, color=col, smooth_shading=True, specular=0.5)

    sargs = dict(n_labels=0)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if dist_mat[i][j] != 0:
                col_i = mcolors.to_rgb(color_dict_rgb.get(symbols[i], 'gray'))
                col_j = mcolors.to_rgb(color_dict_rgb.get(symbols[j], 'gray'))
                cmap_bond = ListedColormap([col_i, col_j])

                points = np.array([coordinates[i], coordinates[j]])
                lines = np.array([2, 0, 1])
                bond_poly = pv.PolyData(points, lines=lines)
                bond_poly.point_data['bond_scalars'] = np.arange(2)

                tube = bond_poly.tube(radius=0.15)
                plotter.add_mesh(tube, scalars='bond_scalars', cmap=cmap_bond, 
                                 smooth_shading=True, show_scalar_bar=False)

def plot_density_isosurfaces(density_values: np.ndarray, iso_value: float, 
                             grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray, 
                             plotter: pv.Plotter, opacity: float = 0.3) -> None:
    grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
    grid.point_data['rho_diff'] = density_values.flatten(order='F')

    iso_att = grid.contour([iso_value], scalars='rho_diff')
    iso_det = grid.contour([-iso_value], scalars='rho_diff')

    plotter.add_mesh(iso_att, color='red', smooth_shading=True, metallic=0.1, roughness=0.3,  
                     specular=1, specular_power=15, opacity=opacity)
    plotter.add_mesh(iso_det, color='darkblue', smooth_shading=True, metallic=0.1, roughness=0.3,  
                     specular=1, specular_power=15, opacity=opacity)

def generate_video(data_dir: Optional[str] = None, res: int = 60, 
                   iso_val: float = 0.002, fps: int = 8, opacity: float = 0.5,
                   view: str = 'iso', interactive_camera: bool = False) -> None:
    
    target_path = os.path.abspath(data_dir if data_dir else os.getcwd())
    name, frames, states = detect_trajectory_info(target_path)
    s_low, s_high = states[0], states[1]

    frames_dir = os.path.join(target_path, "_frames_tmp")
    os.makedirs(frames_dir, exist_ok=True)

    # 1. Compute Global Bounding Box across the entire trajectory
    print("Calculating optimal bounding grid...")
    f_ref = os.path.join(target_path, f"{name}_{frames[0]}_{s_low}.molden")
    atoms_ref, _, _ = parse_molden_file(f_ref)
    ref_coords = np.array([[a['x'], a['y'], a['z']] for a in atoms_ref])
    symbols = [a['label'][0] for a in atoms_ref]
    dist_mat = distance_matrix(ref_coords, len(atoms_ref))
    
    all_aligned_coords = []
    for i in frames:
        f_low_test = os.path.join(target_path, f"{name}_{i}_{s_low}.molden")
        if not os.path.exists(f_low_test):
            continue
        atoms_i, _, _ = parse_molden_file(f_low_test)
        curr_coords = np.array([[a['x'], a['y'], a['z']] for a in atoms_i])
        R, mu_ref, mu_tgt = get_alignment_transform(ref_coords, curr_coords)
        aligned = np.dot(curr_coords - mu_tgt, R.T) + mu_ref
        all_aligned_coords.append(aligned)
        
    all_aligned_coords = np.vstack(all_aligned_coords)
    buffer = 5.0
    min_xyz = all_aligned_coords.min(axis=0) - buffer
    max_xyz = all_aligned_coords.max(axis=0) + buffer
    x_g, y_g, z_g = [np.linspace(min_xyz[j], max_xyz[j], res) for j in range(3)]
    grid_x, grid_y, grid_z = np.meshgrid(x_g, y_g, z_g)
    ref_pts = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T

    # 2. Interactive Camera Setup (If requested)
    custom_camera = None
    if interactive_camera:
        print("\n--- INTERACTIVE CAMERA SETUP ---")
        print("1. An interactive 3D window will now open.")
        print("2. Use your mouse to rotate and zoom to the perfect angle.")
        print("3. Close the window to begin rendering the video from that angle.")
        
        preview = pv.Plotter(window_size=[1280, 720])
        preview.set_background("white")
        
        f_high_prev = os.path.join(target_path, f"{name}_{frames[0]}_{s_high}.molden")
        _, gtos_prev, mos_low_prev = parse_molden_file(f_ref)
        _, _, mos_high_prev = parse_molden_file(f_high_prev)
        
        D_diff_prev = compute_density_matrix([m['coefficients'] for m in mos_high_prev], [m['occupancy'] for m in mos_high_prev]) - \
                      compute_density_matrix([m['coefficients'] for m in mos_low_prev], [m['occupancy'] for m in mos_low_prev])
        
        curr_coords_prev = np.array([[a['x'], a['y'], a['z']] for a in atoms_ref])
        R_prev, mu_ref_temp, mu_tgt_temp = get_alignment_transform(ref_coords, curr_coords_prev)
        local_pts_prev = np.dot(ref_pts - mu_ref_temp, R_prev) + mu_tgt_temp
        
        ao_vals_prev, _ = compute_atomic_orbitals(local_pts_prev, atoms_ref, gtos_prev)
        rho_diff_prev = np.einsum('pi,ij,pj->p', ao_vals_prev, D_diff_prev, ao_vals_prev)

        plot3D(ref_coords, len(atoms_ref), symbols, dist_mat, preview)
        plot_density_isosurfaces(rho_diff_prev, iso_val, grid_x, grid_y, grid_z, preview, opacity)
        
        preview.camera_position = view
        preview.show()
        custom_camera = preview.camera.copy()

    # 3. Setup High-Fidelity Off-Screen Plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.enable_anti_aliasing('ssaa')
    plotter.enable_shadows()
    plotter.renderer.SetUseDepthPeeling(True)
    plotter.renderer.SetMaximumNumberOfPeels(200)
    plotter.renderer.SetOcclusionRatio(0.1)
    
    plotter.add_light(pv.Light(position=(1, 1, 1), intensity=0.9))
    plotter.add_light(pv.Light(position=(-1, -1, 1), intensity=0.6))
    plotter.add_light(pv.Light(position=(0, 0, 1), intensity=0.4))
    plotter.set_background("white")

    # 4. Main Rendering Loop with tqdm
    print("\nStarting frame generation...")
    cam = None
    
    for i in tqdm(frames, desc="Rendering Frames", unit="frame"):
        f_low = os.path.join(target_path, f"{name}_{i}_{s_low}.molden")
        f_high = os.path.join(target_path, f"{name}_{i}_{s_high}.molden")
        
        # Missing frame check
        if not (os.path.exists(f_low) and os.path.exists(f_high)):
            continue
            
        plotter.clear()
        
        atoms, gtos, mos_low = parse_molden_file(f_low)
        _, _, mos_high = parse_molden_file(f_high)
        current_coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])

        R, mu_ref_tgt, mu_tgt_tgt = get_alignment_transform(ref_coords, current_coords)
        local_pts = np.dot(ref_pts - mu_ref_tgt, R) + mu_tgt_tgt
        aligned_current_coords = np.dot(current_coords - mu_tgt_tgt, R) + mu_ref_tgt

        D_diff = compute_density_matrix([m['coefficients'] for m in mos_high], [m['occupancy'] for m in mos_high]) - \
                 compute_density_matrix([m['coefficients'] for m in mos_low], [m['occupancy'] for m in mos_low])
        
        ao_vals, _ = compute_atomic_orbitals(local_pts, atoms, gtos)
        rho_diff = np.einsum('pi,ij,pj->p', ao_vals, D_diff, ao_vals)

        current_dist_mat = distance_matrix(aligned_current_coords, len(atoms))
        plot3D(aligned_current_coords, len(atoms), symbols, current_dist_mat, plotter)
        plot_density_isosurfaces(rho_diff, iso_val, grid_x, grid_y, grid_z, plotter, opacity)

        if cam is None:
            if custom_camera is not None:
                plotter.camera = custom_camera
            else:
                plotter.camera_position = view
                plotter.reset_camera()
                plotter.camera.zoom(0.8)
            cam = plotter.camera.copy()
        else:
            plotter.camera = cam

        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        plotter.screenshot(frame_path)

    # 5. FFMPEG Encoding
    print(f"\nEncoding high-quality MP4 at {fps} FPS...")
    output_file = os.path.join(target_path, f"{name}_{s_high}_to_{s_low}_dynamics.mp4")
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))

    if not frame_files:
        print("Error: No frames were generated. Video encoding aborted.")
        return

    writer = imageio.get_writer(output_file, fps=fps, codec='libx264', macro_block_size=None)
    for f in tqdm(frame_files, desc="Encoding Video", unit="frame"):
        writer.append_data(imageio.imread(f))
    writer.close()

    # Graceful cleanup
    try:
        shutil.rmtree(frames_dir)
    except Exception as e:
        print(f"Warning: Could not delete temporary frame directory: {e}")

    print(f"Success! Video created: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render quantum attachment/detachment trajectories to MP4.")
    parser.add_argument("directory", nargs='?', default=os.getcwd(), help="Target directory containing .molden files")
    parser.add_argument("--res", type=int, default=60, help="Resolution of the 3D sampling grid")
    parser.add_argument("--iso", type=float, default=0.002, help="Cutoff value for the 3D density isosurfaces")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for output video")
    parser.add_argument("--opacity", type=float, default=0.5, help="Opacity level of the density clouds (0.0 to 1.0)")
    parser.add_argument("--view", type=str, default="iso", choices=["iso", "xy", "xz", "yz"], help="Standard camera plane (defaults to 3D isometric)")
    parser.add_argument("--interactive", action="store_true", help="Open a window to manually set the exact camera angle before rendering")
    
    args = parser.parse_args()
    generate_video(
        args.directory, res=args.res, iso_val=args.iso, fps=args.fps, 
        opacity=args.opacity, view=args.view, interactive_camera=args.interactive
    )