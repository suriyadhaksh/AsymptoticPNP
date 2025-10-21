import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from prepare_dataset import extract_data

import jax
import jax.numpy as jnp
from jaxopt import AndersonAcceleration

sys.path.append(os.path.join(os.path.dirname(__file__), "/home/suriya/Mani_Grp/PerturbBCNet/include"))
matplotlib.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
matplotlib.rcParams.update({'text.usetex': 'true'})

import projection as proj

stop_at = 10000

# database_dir = '/home/suriya/Mani_Grp/DATA_set/analytical_clusterting/sin_func_n4/raw_data'
# extract_data(database_dir)

def alpha_inv_jax(Co0, phio0, phi_applied, epsilon):
    # Safe Co0 and denominator check
    Co0 = jnp.where(Co0 <= 0, 1e-16, Co0)
    delta_phi = phi_applied - phio0
    delta_phi = jnp.where(jnp.abs(delta_phi) <= 1e-16, jnp.sign(delta_phi) * 1e-16 + 1e-16, delta_phi)

    sqrt_2 = jnp.sqrt(2.0)
    sinh_term = jnp.sinh(delta_phi / 4.0)
    cosh_term = jnp.cosh(delta_phi / 4.0)
    tanh_term = jnp.tanh(delta_phi / 4.0)

    A11 = 2 * sqrt_2 * epsilon * sinh_term**2 / jnp.sqrt(Co0)
    A12 = -2 * sqrt_2 * jnp.sqrt(Co0) * epsilon * cosh_term * sinh_term
    A21 = -2 * sqrt_2 * epsilon * sinh_term**2 * tanh_term / jnp.sqrt(Co0)
    A22 = (2 * sqrt_2 * jnp.sqrt(Co0) * epsilon * sinh_term**2
           + sqrt_2 * jnp.sqrt(Co0) * epsilon * tanh_term**2)

    determinant = A11 * A22 - A21 * A12
    determinant = jnp.where(jnp.abs(determinant) < 1e-16,
                            jnp.sign(determinant) * 1e-16 + 1e-16,
                            determinant)

    alpha_inv_mat = jnp.array([
        [A22 / determinant, -A21 / determinant],
        [-A12 / determinant, A11 / determinant]
    ])

    return alpha_inv_mat

def get_A_matrix_jax(sol_vec_flat, dt, dx, epsilon):
    N = sol_vec_flat.shape[0] // 2
    A = jnp.zeros((2 * N, 2 * N))

    alpha = dt / (dx ** 2) # this alpha is different from the alpha matrix

    Co0, Co1, phio0, phio1 = sol_vec_flat[0], sol_vec_flat[-2], sol_vec_flat[1], sol_vec_flat[-1]
    alpha_inv_0 = alpha_inv_jax(Co0, phio0, -1, epsilon) 
    alpha_inv_1 = alpha_inv_jax(Co1, phio1, 1, epsilon) 

    k10 = alpha_inv_0[0, 0] * dt / dx
    k20 = alpha_inv_0[0, 1] * Co0 * dt / dx
    k30 = alpha_inv_0[1, 0] * dt / dx
    k40 = alpha_inv_0[1, 1] * Co0 * dt / dx

    # row 0 corresponds to Co0
    A = A.at[0, 0].set(k10+1)
    A = A.at[0, 1].set(k20)
    A = A.at[0, 2].set(-k10)
    A = A.at[0, 3].set(-k20)

    # row 1 corresponds to phio0
    A = A.at[1, 0].set(k30)
    A = A.at[1, 1].set(k40+1)
    A = A.at[1, 2].set(-k30)
    A = A.at[1, 3].set(-k40)

    k11 = - alpha_inv_1[0, 0] * dt / dx
    k21 = - alpha_inv_1[0, 1] * Co1 * dt / dx
    k31 = - alpha_inv_1[1, 0] * dt / dx
    k41 = - alpha_inv_1[1, 1] * Co1 * dt / dx

    # row -2 corresponds to Co1
    A = A.at[-2, -4].set(k11)
    A = A.at[-2, -3].set(k21)
    A = A.at[-2, -2].set(1 - k11)
    A = A.at[-2, -1].set(-k21)

    # row -1 corresponds to phio1
    A = A.at[-1, -4].set(k31)
    A = A.at[-1, -3].set(k41)
    A = A.at[-1, -2].set(-k31)
    A = A.at[-1, -1].set(1 - k41)
    
    def body_fun(i, A):
        A = A.at[2 * i, 2 * (i - 1)].set(-alpha)
        A = A.at[2 * i, 2 * i].set(1 + 2 * alpha)
        A = A.at[2 * i, 2 * (i + 1)].set(-alpha)

        C_j   = sol_vec_flat[2 * i]
        C_jp1 = sol_vec_flat[2 * (i + 1)]
        C_jm1 = sol_vec_flat[2 * (i - 1)]

        d_jp1 = 0.5 * (C_j + C_jp1)
        d_jm1 = 0.5 * (C_j + C_jm1)
        d_j   = - (d_jp1 + d_jm1)

        A = A.at[2 * i + 1, 2 * (i - 1) + 1].set(d_jp1)
        A = A.at[2 * i + 1, 2 * i + 1].set(d_j)
        A = A.at[2 * i + 1, 2 * (i + 1) + 1].set(d_jm1)

        return A

    A = jax.lax.fori_loop(1, N - 1, body_fun, A)
    return A


@jax.jit
def outer_solve_anderson(Co_prev_vec, phio_prev_vec, epsilon, tol=1e-8, max_iter=100):
    N = Co_prev_vec.shape[0]
    Co0, Co1, phio0, phio1 = Co_prev_vec[0], Co_prev_vec[-1], phio_prev_vec[0], phio_prev_vec[-1]
    # Initial guess
    phio_init = jnp.linspace(phio0, phio1, N)
    sol_vec = jnp.stack([Co_prev_vec, phio_init], axis=1)
    sol_vec_flat = sol_vec.flatten()

    # b_vec setup
    b_vec = jnp.zeros((N, 2)).at[:, 0].set(Co_prev_vec)
    b_vec = b_vec.at[0, 0].set(Co0)
    b_vec = b_vec.at[-1, 0].set(Co1)
    b_vec = b_vec.at[0, 1].set(phio0)
    b_vec = b_vec.at[-1, 1].set(phio1)
    b_vec_flat = b_vec.flatten()

    dt = 20 * epsilon / 2000
    dx = 1.0 / (N - 1)

    # Define fixed-point function: x_next = A(x)⁻¹ @ b
    def fixed_point_fn(sol_vec_flat):
        A = get_A_matrix_jax(sol_vec_flat, dt, dx, epsilon)
        return jnp.linalg.solve(A, b_vec_flat)

    solver = AndersonAcceleration(fixed_point_fn, maxiter=max_iter, tol=tol)
    result = solver.run(sol_vec_flat)

    sol_vec_flat_final = result.params
    Co_prev = sol_vec_flat_final[0::2]
    phio_prev = sol_vec_flat_final[1::2]

    return Co_prev, phio_prev

# main function
if __name__ == "__main__":

    # Data setup
    data_base_dir = "./"
    start_at = 1
    no_of_trials  = 17

    # Find maximum number of samples across all trials
    N_sample_max = 0
    epsilon_vec  = []
    for k in range(start_at, no_of_trials + 1):
        data = np.load(os.path.join(data_base_dir, f"trial_{k:02d}.npz"))
        if k == 1:
            time_vec = data['t_vec']
            N_sample_max = min( data['t_vec'].shape[0], stop_at)
        epsilon_vec.append(data['epsilon'])
    epsilon_vec = np.array(epsilon_vec)

    # Pre‐allocate error grids
    C1_log_err   = np.full((no_of_trials, N_sample_max), np.nan)
    C2_log_err   = np.full((no_of_trials, N_sample_max), np.nan)
    phi_log_err  = np.full((no_of_trials, N_sample_max), np.nan)
    
    # Loop through trials and samples
    for k in range(start_at, no_of_trials + 1):
        print(f"[INFO] Running trial {k:02d}")
        data = np.load(os.path.join(data_base_dir, f"trial_{k:02d}.npz"))
        eps  = data['epsilon']
        inputs  = data['input_data']
        outputs = data['output_data']
        N_samp = min(inputs.shape[0], stop_at) # debug

        for i in range(N_samp):
            print(f"[INFO] Sample {i+1}/{N_samp} for trial {k:02d}")
            C1_in, C2_in, phi_in = inputs[i]
            C1_out, C2_out, phi_out = outputs[i]

            # project to (Co_prev, phio_prev) and (Co_true, phio_true)
            Co_prev_vec, phio_prev_vec = proj.projection(C1_in, C2_in, phi_in, eps)
            

            # Solve the outer problem
            Co_vec, phio_vec = outer_solve_anderson(Co_prev_vec, phio_prev_vec, eps)

            C1_next_vec, C2_next_vec, phi_next_vec = proj.inverse_projection(Co_vec, phio_vec, eps)


            C1_loss = np.linalg.norm(C1_next_vec - C1_out, ord=2)
            C1_norm = np.linalg.norm(C1_out, ord=2)

            C2_loss = np.linalg.norm(C2_next_vec - C2_out, ord=2)
            C2_norm = np.linalg.norm(C2_out, ord=2)

            phi_loss = np.linalg.norm(phi_next_vec - phi_out, ord=2)
        
        
            # L2 error at boundaries
            if not (np.isinf(C1_loss) or np.isinf(C2_loss) or np.isinf(phi_loss)):
                C1_rel_loss, C2_rel_loss, phio_loss = C1_loss/C1_norm, C2_loss/C2_norm, phi_loss / np.sqrt(phio_vec.shape[0])
                C1_log_err[k-1, i]  = np.log10(max(np.abs(C1_rel_loss),1E-8))
                C2_log_err[k-1, i]  = np.log10(max(np.abs(C2_rel_loss),1E-8))
                phi_log_err[k-1, i] = np.log10(max(np.abs(phio_loss), 1E-8))
                

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
    
    x = np.arange(N_sample_max)   # sample indices
    y = epsilon_vec               # true epsilon values
    X, Y = np.meshgrid(x, y)

    C1_err_max, C1_err_min = int(C1_err_max), int(C1_err_min)
    C2_err_max, C2_einput_datarr_min = int(C2_err_max), int(C2_err_min)
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
    plt.savefig(f"perturbation_outer_full_implicit_composite_solution_loss.png", dpi=300)
    plt.show()
    plt.close()