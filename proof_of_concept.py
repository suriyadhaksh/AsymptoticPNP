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

      
    # check error limits
    C1_err_max, C1_err_min = np.nanmax(C1_log_err), np.nanmin(C1_log_err)
    C2_err_max, C2_err_min = np.nanmax(C2_log_err), np.nanmin(C2_log_err)
    phi_err_max, phi_err_min = np.nanmax(phi_log_err), np.nanmin(phi_log_err)

    print(f"[INFO] C1 error range in base 10: {C1_err_min} to {C1_err_max}")
    print(f"[INFO] C2 error range in base 10: {C2_err_min} to {C2_err_max}")
    print(f"[INFO] phi error range in base 10: {phi_err_min} to {phi_err_max}")

    # Round the error limits to the nearest integer
    C1_err_min, C1_err_max = np.ceil(C1_err_min), np.ceil(C1_err_max)
    C2_err_min, C2_err_max = np.ceil(C2_err_min), np.ceil(C2_err_max)
    phi_err_min, phi_err_max = np.ceil(phi_err_min), np.ceil(phi_err_max)

    # ---- contour‐plot ----
    vmin = max(np.nanmin([C1_log_err, C2_log_err, phi_log_err]), -8.0) # ensure vmax is not too large
    vmax = min(np.nanmax([C1_log_err, C2_log_err, phi_log_err]), 0.0) # ensure vmax is not too large

    # set floor and ceiling for contour levels
    vmin = np.ceil(vmin)
    vmax = np.ceil(vmax)

    vmin = -7
    vmax = 0
    
    x = np.linspace(0, 20, no_of_time_samples)  # sample indices
    y = epsilon_vec               # true epsilon values
    X, Y = np.meshgrid(x, y)

    C1_err_max, C1_err_min = int(C1_err_max), int(C1_err_min)
    C2_err_max, C2_err_min = int(C2_err_max), int(C2_err_min)
    phi_err_max, phi_err_min = int(phi_err_max), int(phi_err_min)
    

    fig, axes = plt.subplots(1, 3, figsize=(15,5), constrained_layout=True)
    titles =  [ '$C_1$ relative field loss : min, max ' + fr'$ = 10^{{{C1_err_min}}}, 10^{{{C1_err_max}}} $',
                '$C_2$ relative field loss : min, max ' + fr'$ = 10^{{{C2_err_min}}}, 10^{{{C2_err_max}}} $',
                '$\phi$ mean field loss : min, max ' + fr'$ = 10^{{{phi_err_min}}}, 10^{{{phi_err_max}}} $'
              ]

    datasets = [C1_log_err, C2_log_err, phi_log_err]

    levels = np.linspace(vmin, vmax, 40)

    number_of_ticks = vmax + 1 - vmin
    ticks = np.arange(vmin, vmax + 1)
    tick_labels = [f"$10^{{{int(tick)}}}$" for tick in ticks]

    for ax, data, title in zip(axes, datasets, titles):
        cf = ax.contourf(
            X, Y, data,
            levels=levels,
            cmap='RdYlGn_r',
            extend='both'
        )
        ax.set_title(title)
        ax.set_xlabel("$t/\\epsilon$")
        ax.set_yscale('log')

        
        # set y-ticks only on the first axis
        if ax is axes[0]:
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator())
            ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(
                base=10, subs=np.arange(1,10)*0.1))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    axes[0].set_ylabel(r"$\epsilon$")
    # shared colorbar
    cbar = fig.colorbar(cf, ax=axes, pad=0.01)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    plt.savefig(f"perturbation_test_2.png", dpi=300)
    plt.show()
    plt.close()






