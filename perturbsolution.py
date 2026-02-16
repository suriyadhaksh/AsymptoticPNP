import jax
import jax.numpy as jnp
from jaxopt import AndersonAcceleration

from functools import partial
import projection as proj

# @partial(jax.jit, static_argnames=('phi_applied','epsilon'))
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

# @partial(jax.jit, static_argnames=('dt', 'dx', 'epsilon'))
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


# @partial(jax.jit, static_argnames=('epsilon','dt', 'dx', 'tol', 'max_iter'))
def outer_solve_anderson(Co0, Co1, phio0, phio1, Co_prev_vec, phio_prev_vec, epsilon, X_vec, dt, dx, tol=1e-8, max_iter=100):
    N = Co_prev_vec.shape[0]
    
    # Initial guess
    phio_init = phio_prev_vec.at[0].set(phio0).at[-1].set(phio1)
    sol_vec = jnp.stack([Co_prev_vec, phio_init], axis=1)
    sol_vec_flat = sol_vec.flatten()

    # b_vec setup
    b_vec = jnp.zeros((N, 2)).at[:, 0].set(Co_prev_vec)
    b_vec = b_vec.at[0, 0].set(Co0)
    b_vec = b_vec.at[-1, 0].set(Co1)
    b_vec = b_vec.at[0, 1].set(phio0)
    b_vec = b_vec.at[-1, 1].set(phio1)
    b_vec_flat = b_vec.flatten()

    # dt = 20 * epsilon / 2000
    # dx = 1.0 / (N - 1)

    # Define fixed-point function: x_next = A(x)⁻¹ @ b
    def fixed_point_fn(sol_vec_flat):
        A = get_A_matrix_jax(sol_vec_flat, dt, dx, epsilon)
        return jnp.linalg.solve(A, b_vec_flat)

    solver = AndersonAcceleration(fixed_point_fn, maxiter=max_iter, tol=tol)
    result = solver.run(sol_vec_flat)

    print("Converged:", result.state.error < solver.tol)
    print("Iterations:", result.state.iter_num)
    print("Final error:", result.state.error)

    sol_vec_flat_final = result.params
    Co_prev = sol_vec_flat_final[0::2]
    phio_prev = sol_vec_flat_final[1::2]

    return Co_prev, phio_prev 

# @partial(jax.jit, static_argnames=('epsilon','tol', 'max_iter', 'ifInterpol'))
def asymptoticPNPsolve(C1_prev_vec, C2_prev_vec, phi_prev_vec, epsilon, phi_left, phi_right, X_vec, dt, tol=1e-8, max_iter=100, ifInterpol=False):
    N = C1_prev_vec.shape[0]
    X_vec_ = jnp.copy(X_vec)
    
    # Project to outer variables
    #  print("Projection step...")
    Co_prev_vec, phio_prev_vec = proj.projection(C1_prev_vec, C2_prev_vec, phi_prev_vec, epsilon, X_vec, phi_left, phi_right)

    if ifInterpol:
        N_sample = 200
        X_sample_vec = jnp.linspace(0.0, 1.0, N_sample)

        Co_prev_vec = jnp.interp(X_sample_vec, X_vec, Co_prev_vec)
        phio_prev_vec = jnp.interp(X_sample_vec, X_vec, phio_prev_vec)
        X_vec_ = X_sample_vec
        
    dx = X_vec_[1] - X_vec_[0]

    # print("Outer solve with Anderson acceleration...")
    Co_vec, phio_vec = outer_solve_anderson(
        Co_prev_vec[0], Co_prev_vec[-1],
        phio_prev_vec[0], phio_prev_vec[-1],
        Co_prev_vec, phio_prev_vec,
        epsilon, X_vec_, dt, dx,
        tol, max_iter
    )

    if ifInterpol:
        Co_vec = jnp.interp(X_vec, X_vec_, Co_vec)
        phio_vec = jnp.interp(X_vec, X_vec_, phio_vec)

    # Inverse project to inner variables
    # print("Inverse projection step...")
    C1_vec, C2_vec, phi_vec = proj.inverse_projection(Co_vec, phio_vec, epsilon, X_vec, phi_left, phi_right)

    return C1_vec, C2_vec, phi_vec
