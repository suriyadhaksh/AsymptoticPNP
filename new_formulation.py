import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from perturbsolution import asymptoticPNPsolve


# main function
if __name__ == "__main__":

    # ============================================
    # USER SETTINGS
    # ============================================
    selected_trial = 8  # Change this to select which trial to analyze
    if_smaller_dt = True  # Allow smaller time steps
    # ============================================

    # Data setup
    data_base_dir = "./"
    
    # Load the selected trial
    print(f"[INFO] Loading trial {selected_trial}...")
    data = np.load(os.path.join(data_base_dir, f"trial_{selected_trial:02d}.npz"))
    
    X_vec = data['x_vec']
    time_vec = data['t_vec']
    epsilon = data['epsilon']
    inputs = data['input_data']
    outputs = data['output_data']
    
    dt = time_vec[1] - time_vec[0]
    total_steps = len(time_vec)
    n_spatial = len(X_vec)
    
    print(f"[INFO] Trial {selected_trial}: epsilon = {epsilon}")
    print(f"[INFO] Number of time steps: {total_steps}")
    print(f"[INFO] Number of spatial points: {n_spatial}")
    print(f"[INFO] dt = {dt}")
    
    # Pre-allocate arrays for asymptotic solution
    C1_asymp_all = np.zeros((total_steps, n_spatial))
    C2_asymp_all = np.zeros((total_steps, n_spatial))
    phi_asymp_all = np.zeros((total_steps, n_spatial))
    
    # Pre-allocate error arrays
    C1_l2_error = np.zeros(total_steps)
    C2_l2_error = np.zeros(total_steps)
    phi_l2_error = np.zeros(total_steps)
    
    # Loop over all time steps
    for time_index in range(int(total_steps/10)):
        print(f"[INFO] Processing time step {time_index+1}/{total_steps} (t = {time_vec[time_index]:.5e})...")
        
        C1_in, C2_in, phi_in = inputs[time_index]
        C1_out, C2_out, phi_out = outputs[time_index]
        
        if not if_smaller_dt:
            # Asymptotic PNP solve
            C1_asymp, C2_asymp, phi_asymp = asymptoticPNPsolve(
                C1_in, C2_in, phi_in,
                epsilon,
                -1.0, 1.0,
                X_vec,
                dt,
                tol=1e-8,
                max_iter=1000,
                ifInterpol=True
            )
        else:
            dt_temp = dt / 8.0
            t_current = 0.0
            C1_asymp, C2_asymp, phi_asymp = C1_in, C2_in, phi_in
            while t_current < dt:
                C1_asymp, C2_asymp, phi_asymp = asymptoticPNPsolve(
                    C1_asymp, C2_asymp, phi_asymp,
                    epsilon,
                    -1.0, 1.0,
                    X_vec,
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
        C1_l2_error[time_index] = np.linalg.norm(C1_asymp - C1_out, ord=2) / np.linalg.norm(C1_out, ord=2)
        C2_l2_error[time_index] = np.linalg.norm(C2_asymp - C2_out, ord=2) / np.linalg.norm(C2_out, ord=2)
        phi_l2_error[time_index] = np.linalg.norm(phi_asymp - phi_out, ord=2) / np.sqrt(phi_asymp.shape[0])
    
    print("[INFO] All time steps processed. Generating plots...")
    
    # Extract DNS solutions
    C1_dns_all = outputs[:, 0, :]
    C2_dns_all = outputs[:, 1, :]
    phi_dns_all = outputs[:, 2, :]
    
    # ============================================
    # PLOT 1: DNS vs Perturbation XT plots
    # ============================================
    
    # Create meshgrid for contour plots
    T, X = np.meshgrid(time_vec, X_vec, indexing='ij')
    
    # Determine color limits from DNS data only
    C1_vmin = np.min(C1_dns_all)
    C1_vmax = np.max(C1_dns_all)
    C2_vmin = np.min(C2_dns_all)
    C2_vmax = np.max(C2_dns_all)
    
    # Use same limits for C1 and C2 for comparison
    C_vmin = min(C1_vmin, C2_vmin)
    C_vmax = max(C1_vmax, C2_vmax)
    
    # Separate limits for phi
    phi_vmin = np.min(phi_dns_all)
    phi_vmax = np.max(phi_dns_all)
    
    print(f"[INFO] C1/C2 color range: [{C_vmin:.4f}, {C_vmax:.4f}]")
    print(f"[INFO] phi color range: [{phi_vmin:.4f}, {phi_vmax:.4f}]")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    
    # Row 1: DNS
    cf1 = axes[0, 0].contourf(T, X, C1_dns_all, levels=50, cmap='viridis', vmin=C_vmin, vmax=C_vmax)
    axes[0, 0].set_title(f'DNS: $C_1$ (Trial {selected_trial}, $\\epsilon={epsilon:.4f}$)')
    axes[0, 0].set_ylabel('x')
    axes[0, 0].set_xlabel('t')
    
    cf2 = axes[0, 1].contourf(T, X, C2_dns_all, levels=50, cmap='viridis', vmin=C_vmin, vmax=C_vmax)
    axes[0, 1].set_title(f'DNS: $C_2$ (Trial {selected_trial}, $\\epsilon={epsilon:.4f}$)')
    axes[0, 1].set_ylabel('x')
    axes[0, 1].set_xlabel('t')
    
    cf3 = axes[0, 2].contourf(T, X, phi_dns_all, levels=50, cmap='plasma', vmin=phi_vmin, vmax=phi_vmax)
    axes[0, 2].set_title(f'DNS: $\\phi$ (Trial {selected_trial}, $\\epsilon={epsilon:.4f}$)')
    axes[0, 2].set_ylabel('x')
    axes[0, 2].set_xlabel('t')
    
    # Row 2: Asymptotic
    cf4 = axes[1, 0].contourf(T, X, C1_asymp_all, levels=50, cmap='viridis', vmin=C_vmin, vmax=C_vmax)
    axes[1, 0].set_title(f'Asymptotic: $C_1$')
    axes[1, 0].set_ylabel('x')
    axes[1, 0].set_xlabel('t')
    
    cf5 = axes[1, 1].contourf(T, X, C2_asymp_all, levels=50, cmap='viridis', vmin=C_vmin, vmax=C_vmax)
    axes[1, 1].set_title(f'Asymptotic: $C_2$')
    axes[1, 1].set_ylabel('x')
    axes[1, 1].set_xlabel('t')
    
    cf6 = axes[1, 2].contourf(T, X, phi_asymp_all, levels=50, cmap='plasma', vmin=phi_vmin, vmax=phi_vmax)
    axes[1, 2].set_title(f'Asymptotic: $\\phi$')
    axes[1, 2].set_ylabel('x')
    axes[1, 2].set_xlabel('t')
    
    # Add colorbars
    cbar1 = fig.colorbar(cf1, ax=axes[:, 0:2].ravel().tolist(), pad=0.01, aspect=30)
    cbar1.set_label('$C_1, C_2$')
    
    cbar2 = fig.colorbar(cf3, ax=axes[:, 2].ravel().tolist(), pad=0.01, aspect=30)
    cbar2.set_label('$\\phi$')
    
    plt.savefig(f"trial_{selected_trial:02d}_xt_comparison.png", dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved: trial_{selected_trial:02d}_xt_comparison.png")
    plt.show()
    plt.close()
    
    # ============================================
    # PLOT 2: L2 Error vs Time (log scale)
    # ============================================
    
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    ax.semilogy(time_vec, C1_l2_error, label='$C_1$ relative error', linewidth=2)
    ax.semilogy(time_vec, C2_l2_error, label='$C_2$ relative error', linewidth=2)
    ax.semilogy(time_vec, phi_l2_error, label='$\\phi$ mean error', linewidth=2)
    
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('L2 Error (log scale)', fontsize=12)
    ax.set_title(f'L2 Error vs Time (Trial {selected_trial}, $\\epsilon={epsilon:.4f}$)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    
    plt.savefig(f"trial_{selected_trial:02d}_l2_error_vs_time.png", dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved: trial_{selected_trial:02d}_l2_error_vs_time.png")
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