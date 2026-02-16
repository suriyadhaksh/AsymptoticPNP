import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from perturbsolution import asymptoticPNPsolve


# main functiton
if __name__ == "__main__":

    # Allow smaller time steps
    if_smaller_dt = False

    # Data setup
    data_base_dir = "./"
    start_at = 1
    end_at = 17
    skip_trials = 5
    no_of_trials = 0

    trial_index_vec = range(start_at, end_at + 1, skip_trials)

    time_data_percent = 0.01

    # Fix x and y axes limits for plots
    epsilon_vec = []
    time_index_vec = None
    dt = 0.0


    for k in range(start_at, end_at + 1, skip_trials):
        no_of_trials += 1
        data = np.load(os.path.join(data_base_dir, f"trial_{k:02d}.npz"))
        if k == 1:
            time_vec = data['t_vec']
            dt = time_vec[1] - time_vec[0]
            epsilon = data['epsilon']

            total_steps = len(time_vec)
            no_of_time_samples = int(time_data_percent * total_steps)
            time_index_vec = np.linspace(0, total_steps, no_of_time_samples, endpoint=False, dtype=int)

            no_of_time_samples = len(time_index_vec)

        epsilon_vec.append(data['epsilon'])

    epsilon_vec = np.array(epsilon_vec)

    C_max, C_min = 2.7, 0.0
    phi_max, phi_min = -1.0, 1.0
    
    # Loop over trials
    for k, trial_no in enumerate(trial_index_vec):
        print(f"[INFO] Processing trial {trial_no}...")
        data = np.load(os.path.join(data_base_dir, f"trial_{trial_no:02d}.npz"))
        X_vec = data['x_vec']
        eps = data['epsilon']
        inputs = data['input_data']
        outputs = data['output_data']

        x_num_of_points = X_vec.shape[0]
        time_vec_epsilon = np.linspace(0, 20, total_steps)

        X_mesh, T_mesh = np.meshgrid(X_vec, time_vec_epsilon[time_index_vec])

        C1_DNS_grid = np.zeros((no_of_time_samples, x_num_of_points))
        C2_DNS_grid = np.zeros((no_of_time_samples, x_num_of_points))
        phi_DNS_grid = np.zeros((no_of_time_samples, x_num_of_points))

        C1_asymp_grid = np.zeros((no_of_time_samples, x_num_of_points))
        C2_asymp_grid = np.zeros((no_of_time_samples, x_num_of_points))
        phi_asymp_grid = np.zeros((no_of_time_samples, x_num_of_points))
      
        for i, time_index in enumerate(time_index_vec):
            print(f"\n[INFO]  Sample {i+1}/{no_of_time_samples} for trial_{trial_no:02d}...")
            C1_in, C2_in, phi_in = inputs[time_index]
            C1_out, C2_out, phi_out = outputs[time_index]

            if not if_smaller_dt:
                # Asymptotic PNP solve
                C1_asymp, C2_asymp, phi_asymp = asymptoticPNPsolve(
                    C1_in, C2_in, phi_in,
                    eps,
                    -1.0, 1.0,
                    X_vec,
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
                        eps,
                        -1.0, 1.0,
                        X_vec,
                        dt_temp,
                        tol=1e-5,
                        max_iter=1000,
                        ifInterpol=True
                    )

                    t_current += dt_temp
                    print(f"[INTERNAL LOOP] Advanced to t = {t_current:.5e} / {dt:.5e}")

                # Store DNS and asymptotic results for visualization
            C1_DNS_grid[i, :]  = C1_out
            C2_DNS_grid[i, :]  = C2_out
            phi_DNS_grid[i, :] = phi_out

            C1_asymp_grid[i, :]  = C1_asymp
            C2_asymp_grid[i, :]  = C2_asymp
            phi_asymp_grid[i, :] = phi_asymp

        # ----------------------------------------------------------------------
        # After completing all time steps for this trial → plot results
        # ----------------------------------------------------------------------

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        fields = [
            (C1_DNS_grid, C1_asymp_grid, "C₁", C_min, C_max),
            (C2_DNS_grid, C2_asymp_grid, "C₂", C_min, C_max),
            (phi_DNS_grid, phi_asymp_grid, "ϕ", phi_min, phi_max)
        ]

        for col, (dns_grid, asymp_grid, title, vmin, vmax) in enumerate(fields):
            # --- Row 1: DNS ---
            pcm_dns = axes[0, col].pcolormesh(
                X_mesh, T_mesh, dns_grid,
                shading='auto', cmap='viridis', vmin=vmin, vmax=vmax
            )
            cbar_dns = fig.colorbar(pcm_dns, ax=axes[0, col])
            cbar_dns.set_label(f"{title} (DNS)")
            axes[0, col].set_title(f"{title} (DNS)")
            axes[0, col].set_ylabel("Time/$\epsilon$")
            axes[0, col].set_xlabel("x")

            # --- Row 2: Asymptotic ---
            pcm_asymp = axes[1, col].pcolormesh(
                X_mesh, T_mesh, asymp_grid,
                shading='auto', cmap='viridis', vmin=vmin, vmax=vmax
            )
            cbar_asymp = fig.colorbar(pcm_asymp, ax=axes[1, col])
            cbar_asymp.set_label(f"{title} (Asymptotic)")
            axes[1, col].set_title(f"{title} (Asymptotic)")
            axes[1, col].set_ylabel("Time/$\epsilon$")
            axes[1, col].set_xlabel("x")

        fig.suptitle(
            f"Trial {trial_no:02d} — ε = {eps:.2e}",
            fontsize=16, fontweight='bold'
        )

        # Create output directory if not exists
        os.makedirs("figures_dx4", exist_ok=True)

        # Save figure
        out_path = os.path.join("figures_dx4", f"trial_{trial_no:02d}.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"[SAVED] {out_path}")
            
                

        
        





