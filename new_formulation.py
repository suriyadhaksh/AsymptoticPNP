import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import re
import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from perturbsolution import asymptoticPNPsolve
from projection import projection, inverse_projection


def extract_grids(trial_folder_path):
    """
    Extract epsilon, x_vec, t_vec, and solution grids (C1, C2, Phi)
    from a trial directory containing config.txt, data_init.plt, and output_manifest.csv.
    """

    trial_path = Path(trial_folder_path)
    print(f"Opening {trial_folder_path}")

    # ------------------------------------------------------------
    # 1. Read config.txt to extract epsilon (dbLength)
    # ------------------------------------------------------------
    config_path = trial_path / "config.txt"
    cfg_text = config_path.read_text()
    m = re.search(r"dbLength\s*=\s*([0-9.eE+-]+)\s*;", cfg_text)
    if not m:
        raise ValueError(f"dbLength not found in {config_path}")
    epsilon = float(m.group(1))

    # ------------------------------------------------------------
    # 2. Extract x_vec from data_init.plt
    # ------------------------------------------------------------
    data_init_path = trial_path / "data_initial.plt"
    with open(data_init_path, "r") as f:
        lines = f.readlines()

    # find ZONE line
    zone_line = next((i for i, l in enumerate(lines) if l.strip().startswith("ZONE")), None)
    if zone_line is None:
        raise ValueError("ZONE line not found in data_init.plt")

    # extract number of points I=...
    m = re.search(r"I\s*=\s*(\d+)", lines[zone_line])
    npts = int(m.group(1))
    start_idx = zone_line + 1
    end_idx = start_idx + npts
    data = np.loadtxt(lines[start_idx:end_idx])
    x_vec = data[:, 0]

    # ------------------------------------------------------------
    # 3. Extract t_vec and filenames from output_manifest.csv
    # ------------------------------------------------------------
    manifest_path = trial_path / "output_manifest.csv"
    df = pd.read_csv(manifest_path)
    t_vec = df["t"].to_numpy()
    file_names = df["file_name"].to_list()

    # ------------------------------------------------------------
    # 4. Initialize grids
    # ------------------------------------------------------------
    n_t = len(t_vec)
    n_x = len(x_vec)
    C1_grid = np.zeros((n_t, n_x))
    C2_grid = np.zeros((n_t, n_x))
    phi_grid = np.zeros((n_t, n_x))

    # ------------------------------------------------------------
    # 5. Fill grids from each .plt file
    # ------------------------------------------------------------
    for i, fname in enumerate(file_names):
        plt_path = trial_path / fname
        if not plt_path.exists():
            print(f"Warning: missing {fname}")
            continue

        with open(plt_path, "r") as f:
            lines = f.readlines()

        zone_line = next((j for j, l in enumerate(lines) if l.strip().startswith("ZONE")), None)
        if zone_line is None:
            raise ValueError(f"ZONE line not found in {fname}")

        m = re.search(r"I\s*=\s*(\d+)", lines[zone_line])
        npts = int(m.group(1))
        start_idx = zone_line + 1
        end_idx = start_idx + npts

        # Each line: x, C1, C2, Phi
        data = np.loadtxt(lines[start_idx:end_idx])
        C1_grid[i, :] = data[:, 1]
        C2_grid[i, :] = data[:, 2]
        phi_grid[i, :] = data[:, 3]

    # ------------------------------------------------------------
    # 6. Return all extracted arrays
    # ------------------------------------------------------------
    return epsilon, x_vec, t_vec, C1_grid, C2_grid, phi_grid


# main function
if __name__ == "__main__":

    # ============================================
    # USER SETTINGS
    # ============================================n
    selected_trial = 8  # Change this to select which trial to analyze
    if_smaller_dt = True  # Allow smaller time steps
    base_input_dir = "/home/suriya/Mani_Grp/DATA_set/time_clustering/sinN8_func"  # Base directory containing trial folders
    base_output_dir = "./"  # Output directory for figures
    phi_left = -1.0
    phi_right = 1.0
    # ============================================

    # ---------------------------------------------------------
    # Step 1: Import DNS data
    # ---------------------------------------------------------
    trial_name = f"trial_{selected_trial:02d}"
    trial_folder_path = os.path.join(base_input_dir, trial_name)
    
    print(f"\n==============================")
    print(f"🔹 Processing {trial_name}")
    print(f"==============================")
    
    # Initialize data variables
    epsilon = None
    x_vec = None
    t_vec = None
    C1_grid = None
    C2_grid = None
    phi_grid = None
    
    try:
        epsilon, x_vec, t_vec, C1_grid, C2_grid, phi_grid = extract_grids(trial_folder_path)
        print("✓ Grid info extracted successfully")
    except Exception as e:
        print(f"❌ Error processing {trial_name}: {e}\n")
        sys.exit(1)
    
    total_steps = len(t_vec)
    n_spatial = len(x_vec)
    
    print(f"[INFO] Trial {selected_trial}: epsilon = {epsilon}")
    print(f"[INFO] Number of time steps: {total_steps}")
    print(f"[INFO] Number of spatial points: {n_spatial}")
    print(f"[INFO] Time range: [{t_vec[0]:.5e}, {t_vec[-1]:.5e}]")
    
    # Compute dt for each time step (non-uniform)
    dt_vec = np.zeros(total_steps)
    dt_vec[0] = 0.0  # First dt is just t_vec[0] (assuming starting from 0)
    dt_vec[1:] = np.diff(t_vec)
    
    print(f"[INFO] dt range: [{np.min(dt_vec):.5e}, {np.max(dt_vec):.5e}]")
    
    # ---------------------------------------------------------
    # Step 2: Run asymptotic solver for all time steps
    # ---------------------------------------------------------
    C1_asymp_all = np.full((total_steps, n_spatial), np.nan)
    C2_asymp_all = np.full((total_steps, n_spatial), np.nan)
    phi_asymp_all = np.full((total_steps, n_spatial), np.nan)

    # Grids to store outer  solutions
    Co_outer_grid = np.zeros_like(C1_asymp_all)
    phio_outer_grid = np.zeros_like(phi_asymp_all)

    C1_l2_error = np.full(total_steps, np.nan)
    C2_l2_error = np.full(total_steps, np.nan)
    phi_l2_error = np.full(total_steps, np.nan)

    # Initial conditions vectors and scalars
    Co_prev_vec    = None
    phio_prev_vec  = None
    Co0_prev       = None
    Co1_prev       = None
    phio0_prev     = None
    phio1_prev     = None

    mathscrC0_prev = None
    mathscrC1_prev = None
    varrho0_prev    = None
    varrho1_prev    = None

    startPerturbation = False
    enableUniformGrid = True
    
    # Loop over all time steps
    for time_index in range(89): #range(total_steps):
        t_n = t_vec[time_index]
        if startPerturbation:
            print("\n" )

        print(f"[INFO] Processing time step {time_index+1}/{total_steps} (t = {t_vec[time_index]:.5e})...")
        
        if t_n >= epsilon and (not startPerturbation):
            startPerturbation = True
            print(" ---------------------------------------------------------------------------------- ")
            print(f"   Critical time crossed: t = {t_n:.3e} > ε = {epsilon:g}")

            # Get the initial condition for the outer problem at time t_n
            C1_prev_vec  = C1_grid[time_index, :]
            C2_prev_vec  = C2_grid[time_index, :]
            phi_prev_vec = phi_grid[time_index, :]


            Co_prev_vec, phio_prev_vec = projection(
                C1_prev_vec, C2_prev_vec, phi_prev_vec,
                epsilon, x_vec, phi_left, phi_right
            )

            Co0_prev = Co_prev_vec[0]
            Co1_prev = Co_prev_vec[-1]
            phio0_prev = phio_prev_vec[0]
            phio1_prev = phio_prev_vec[-1]

            print(f"   Initial conditions set")
            print(f"   Initial outer BCs: Co0 = {Co0_prev:.3e}, Co1 = {Co1_prev:.3e}, phio0 = {phio0_prev:.3e}, phio1 = {phio1_prev:.3e}")
            # print(f"   Initial perturbation BCs: 𝓒₀ = {mathscrC0_prev:.3e}, 𝓒₁ = {mathscrC1_prev:.3e}, ρ₀ = {varrho0_prev:.3e}, ρ₁ = {varrho1_prev:.3e}")
            print(" --------------------------------------------------------------------------------- \n")


        if startPerturbation:
            # Get DNS solution at this time step
            C1_dns = C1_grid[time_index, :]
            C2_dns = C2_grid[time_index, :]
            phi_dns = phi_grid[time_index, :]
            
            #  Use previous asymptotic solution as input
            C1_in = C1_asymp_all[time_index - 1, :]
            C2_in = C2_asymp_all[time_index - 1, :]
            phi_in = phi_asymp_all[time_index - 1, :]
            
            # Get dt for this time step
            dt = dt_vec[time_index]

            if enableUniformGrid:
                x_uniform = np.linspace(x_vec[0], x_vec[-1], 30)
                Co_prev_vec_uniform = np.interp(x_uniform, x_vec, Co_prev_vec)
                phio_prev_vec_uniform = np.interp(x_uniform, x_vec, phio_prev_vec)
            
            if not if_smaller_dt:
                # Asymptotic PNP solve
                C1_asymp, C2_asymp, phi_asymp = asymptoticPNPsolve(
                    C1_in, C2_in, phi_in,
                    epsilon,
                    phi_left, phi_right,
                    x_vec,
                    dt,
                    tol=1e-8,
                    max_iter=1000,
                    ifInterpol=True
                )
            else:
                dt_temp = dt / 4.0
                t_current = 0.0
                C1_asymp, C2_asymp, phi_asymp = C1_in, C2_in, phi_in
                while t_current < dt:
                    C1_asymp, C2_asymp, phi_asymp = asymptoticPNPsolve(
                        C1_asymp, C2_asymp, phi_asymp,
                        epsilon,
                        phi_left, phi_right,
                        x_vec,
                        dt_temp,
                        tol=1e-8,
                        max_iter=100,
                        ifInterpol=True
                    )
                    t_current += dt_temp
                    print(f"[INTERNAL LOOP] Advanced to t = {t_current:.5e} / {dt:.5e}")

                    # Store asymptotic solution
                    C1_asymp_all[time_index, :] = C1_asymp
                    C2_asymp_all[time_index, :] = C2_asymp
                    phi_asymp_all[time_index, :] = phi_asymp
                    
                    # Compute L2 errors
                    C1_norm = np.linalg.norm(C1_dns, ord=2)
                    C2_norm = np.linalg.norm(C2_dns, ord=2)
                    
                    if C1_norm > 1e-12:
                        C1_l2_error[time_index] = np.linalg.norm(C1_asymp - C1_dns, ord=2) / C1_norm
                    else:
                        C1_l2_error[time_index] = np.linalg.norm(C1_asymp - C1_dns, ord=2)
                    
                    if C2_norm > 1e-12:
                        C2_l2_error[time_index] = np.linalg.norm(C2_asymp - C2_dns, ord=2) / C2_norm
                    else:
                        C2_l2_error[time_index] = np.linalg.norm(C2_asymp - C2_dns, ord=2)
                    
                    phi_l2_error[time_index] = np.linalg.norm(phi_asymp - phi_dns, ord=2) / np.sqrt(phi_asymp.shape[0])
    
    print("[INFO] All time steps processed. Generating plots...")
    
    # ---------------------------------------------------------
    # Step 3: Generate plots
    # ---------------------------------------------------------
    
    # ============================================
    # PLOT 1: DNS vs Perturbation XT plots
    # ============================================
    
    # Add small offset for log safety
    eps_offset = 1e-4
    t_plot = t_vec + eps_offset
    
    # Create meshgrid for contour plots (note: x on horizontal, t on vertical)
    X, T = np.meshgrid(x_vec, t_plot)
    
    # Determine color limits from DNS data only
    C1_vmin = np.min(C1_grid)
    C1_vmax = np.max(C1_grid)
    C2_vmin = np.min(C2_grid)
    C2_vmax = np.max(C2_grid)
    
    # Use same limits for C1 and C2 for comparison
    C_vmin = min(C1_vmin, C2_vmin)
    C_vmax = max(C1_vmax, C2_vmax)
    
    # Separate limits for phi
    phi_vmin = np.min(phi_grid)
    phi_vmax = np.max(phi_grid)
    
    print(f"[INFO] C1/C2 color range: [{C_vmin:.4f}, {C_vmax:.4f}]")
    print(f"[INFO] phi color range: [{phi_vmin:.4f}, {phi_vmax:.4f}]")
    
    fig = plt.figure(figsize=(14, 8))
    
    # Row 1: DNS
    ax11 = plt.subplot(2, 3, 1)
    pcm1 = ax11.pcolormesh(X, T, C1_grid, shading='auto', cmap='viridis', vmin=C_vmin, vmax=C_vmax)
    ax11.set_yscale('log')
    ax11.set_title(f'DNS: $C_1$')
    ax11.set_ylabel('t')
    ax11.axhline(epsilon, color='red', linestyle='--', linewidth=1.5, label=r'$t=\epsilon$')
    
    ax12 = plt.subplot(2, 3, 2)
    pcm2 = ax12.pcolormesh(X, T, C2_grid, shading='auto', cmap='viridis', vmin=C_vmin, vmax=C_vmax)
    ax12.set_yscale('log')
    ax12.set_title(f'DNS: $C_2$')
    ax12.axhline(epsilon, color='red', linestyle='--', linewidth=1.5)
    
    ax13 = plt.subplot(2, 3, 3)
    pcm3 = ax13.pcolormesh(X, T, phi_grid, shading='auto', cmap='plasma', vmin=phi_vmin, vmax=phi_vmax)
    ax13.set_yscale('log')
    ax13.set_title(f'DNS: $\\phi$')
    ax13.axhline(epsilon, color='red', linestyle='--', linewidth=1.5)
    
    # Row 2: Asymptotic
    ax21 = plt.subplot(2, 3, 4)
    pcm4 = ax21.pcolormesh(X, T, C1_asymp_all, shading='auto', cmap='viridis', vmin=C_vmin, vmax=C_vmax)
    ax21.set_yscale('log')
    ax21.set_xlabel('x')
    ax21.set_ylabel('t')
    ax21.set_title(f'Perturbation: $C_1$')
    ax21.axhline(epsilon, color='red', linestyle='--', linewidth=1.5)
    
    ax22 = plt.subplot(2, 3, 5)
    pcm5 = ax22.pcolormesh(X, T, C2_asymp_all, shading='auto', cmap='viridis', vmin=C_vmin, vmax=C_vmax)
    ax22.set_yscale('log')
    ax22.set_xlabel('x')
    ax22.set_title(f'Perturbation: $C_2$')
    ax22.axhline(epsilon, color='red', linestyle='--', linewidth=1.5)
    
    ax23 = plt.subplot(2, 3, 6)
    pcm6 = ax23.pcolormesh(X, T, phi_asymp_all, shading='auto', cmap='plasma', vmin=phi_vmin, vmax=phi_vmax)
    ax23.set_yscale('log')
    ax23.set_xlabel('x')
    ax23.set_title(f'Perturbation: $\\phi$')
    ax23.axhline(epsilon, color='red', linestyle='--', linewidth=1.5)
    
    # Adjust spacing before adding colorbars
    plt.subplots_adjust(
        left=0.06,
        right=0.88,
        bottom=0.08,
        top=0.90,
        wspace=0.25,
        hspace=0.30
    )
    
    # Add colorbars
    cbar1 = plt.colorbar(
        pcm1,
        ax=[ax11, ax12, ax21, ax22],
        fraction=0.035,
        pad=0.02
    )
    cbar1.set_label('Concentration ($C_1, C_2$)')
    
    cbar2 = plt.colorbar(
        pcm3,
        ax=[ax13, ax23],
        fraction=0.035,
        pad=0.02
    )
    cbar2.set_label('Potential ($\\phi$)')
    
    plt.suptitle(
        f'DNS vs Perturbation Comparison (Trial {selected_trial}), '
        f'$\\epsilon = {epsilon:.3e}$',
        fontsize=14
    )
    
    output_path = os.path.join(base_output_dir, f"trial_{selected_trial:02d}_xt_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved: {output_path}")
    plt.show()
    plt.close()
    
    # ============================================
    # PLOT 2: L2 Error vs Time (log scale)
    # ============================================
    
    # Add small offset for log safety
    eps_offset = 1e-4
    t_plot = t_vec + eps_offset
    C1_err_plot = np.abs(C1_l2_error) + eps_offset
    C2_err_plot = np.abs(C2_l2_error) + eps_offset
    phi_err_plot = np.abs(phi_l2_error) + eps_offset
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(t_plot, C1_err_plot, label=r'$C_1$ error', linewidth=2)
    plt.plot(t_plot, C2_err_plot, label=r'$C_2$ error', linewidth=2)
    plt.plot(t_plot, phi_err_plot, label=r'$\phi$ error', linewidth=2)
    
    plt.axvline(
        epsilon,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=r'$t=\epsilon$'
    )
    
    plt.xscale('log')
    plt.xlim(eps_offset, max(t_vec))
    plt.yscale('log')
    
    plt.xlabel('t', fontsize=12)
    plt.ylabel('Error norm', fontsize=12)
    plt.title(f'Error norms vs time ($\\epsilon = {epsilon:.3e}$)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    output_path = os.path.join(base_output_dir, f"trial_{selected_trial:02d}_l2_error_vs_time.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved: {output_path}")
    plt.show()
    plt.close()
    
    print("[INFO] All plots generated successfully!")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"C1 L2 Error - Mean: {np.mean(C1_l2_error):.6e}, Max: {np.max(C1_l2_error):.6e}, Min: {np.min(C1_l2_error):.6e}")
    print(f"C2 L2 Error - Mean: {np.mean(C2_l2_error):.6e}, Max: {np.max(C2_l2_error):.6e}, Min: {np.min(C2_l2_error):.6e}")
    print(f"phi L2 Error - Mean: {np.mean(phi_l2_error):.6e}, Max: {np.max(phi_l2_error):.6e}, Min: {np.min(phi_l2_error):.6e}")
    print("="*60)