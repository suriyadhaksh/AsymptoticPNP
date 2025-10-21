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
    if_smaller_dt = True

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

    # Pre-allocate error arrays
    C1_log_err   = np.full((no_of_trials, no_of_time_samples), np.nan)
    C2_log_err   = np.full((no_of_trials, no_of_time_samples), np.nan)
    phi_log_err  = np.full((no_of_trials, no_of_time_samples), np.nan)

    # Loop over trials
    for k, trial_no in enumerate(trial_index_vec):
        print(f"[INFO] Processing trial {trial_no}...")
        data = np.load(os.path.join(data_base_dir, f"trial_{trial_no:02d}.npz"))
        X_vec = data['x_vec']
        eps = data['epsilon']
        inputs = data['input_data']
        outputs = data['output_data']

        x_num_of_points = X_vec.shape[0]
        time_vec_epsilon = np.linspace(0, 20, no_of_time_samples)

        X_mesh, T_mesh = np.meshgrid(X_vec, time_vec[time_index_vec])

        C1_DNS_grid = np.zeros((no_of_time_samples, x_num_of_points))
        C2_DNS_grid = np.zeros((no_of_time_samples, x_num_of_points))
        phi_DNS_grid = np.zeros((no_of_time_samples, x_num_of_points))


        C1_asymp 


        
        for i, time_index in enumerate(time_index_vec):
            print(f"[INFO]  Sample {i+1}/{no_of_time_samples} for trial_{trial_no:02d}...")
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
                    max_iter=100,
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
                        tol=1e-8,
                        max_iter=100,
                        ifInterpol=True
                    )

                    t_current += dt_temp
                    print(f"[INTERNAL LOOP] Advanced to t = {t_current:.5e} / {dt:.5e}")

            # Compute log10 errors
            C1_loss = np.linalg.norm(C1_asymp - C1_out, ord=2)
            C1_norm = np.linalg.norm(C1_out, ord=2)

            C2_loss = np.linalg.norm(C2_asymp - C2_out, ord=2)
            C2_norm = np.linalg.norm(C2_out, ord=2)

            phi_loss = np.linalg.norm(phi_asymp - phi_out, ord=2)

            # L2 error at boundaries
            if not (np.isinf(C1_loss) or np.isinf(C2_loss) or np.isinf(phi_loss)):
                C1_rel_loss, C2_rel_loss, phi_loss = C1_loss/C1_norm, C2_loss/C2_norm, phi_loss / np.sqrt(phi_asymp.shape[0])
                C1_log_err[k, i]  = np.log10(max(np.abs(C1_rel_loss),1E-8))
                C2_log_err[k, i]  = np.log10(max(np.abs(C2_rel_loss),1E-8))
                phi_log_err[k, i] = np.log10(max(np.abs(phi_loss), 1E-8))

      
    





