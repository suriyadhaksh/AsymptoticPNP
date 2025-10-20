import innersolution as innersol
import jax.numpy as jnp

'''
(Differentiable) projection operator for the Perturbed 1D PNP Blocking electrode problem.
The projector operator (P) projects the composite (or the DNS) solution fields (C1(X), C2(X), phi(X), epsilon) to the outer solution fields (Co(X), phio(X)).
See [../docs/api/projection.md] for details.
'''

def projection(C1_vec, C2_vec, phi_vec, epsilon):

    N = 1000
    if N != C1_vec.size:
        raise ValueError(f'The length of C1_vec must be {N}, but got {C1_vec.size}')
    X_vec = jnp.linspace(0.0, 1.0, N)
    
    phi_left = -1.0
    phi_right = 1.0

    Co0 = 2 * jnp.sqrt(C1_vec[0] * C2_vec[0])
    Co1 = 2 * jnp.sqrt(C1_vec[-1] * C2_vec[-1])
    phio0 = phi_left + 0.5 * jnp.log(C1_vec[0]/C2_vec[0])
    phio1 = phi_right + 0.5 * jnp.log(C1_vec[-1]/C2_vec[-1])

    Co_vec = C1_vec + C2_vec - innersol.Ci_left(Co0, phio0, epsilon) + Co0 - innersol.Ci_right(Co1, phio1, epsilon) + Co1
    phio_vec = phi_vec - innersol.phii_left(Co0, phio0, epsilon) + phio0 - innersol.phii_right(Co1, phio1, epsilon) + phio1

    N_sample = 100
    X_sample_vec = jnp.linspace(0.0, 1.0, N_sample)

    Co_vec_ = jnp.interp(X_sample_vec, X_vec, Co_vec)
    phio_vec_ = jnp.interp(X_sample_vec, X_vec, phio_vec)

    return Co_vec_, phio_vec_

def inverse_projection(Co_vec, phio_vec, epsilon):

    N = 100
    if N != Co_vec.size:
        raise ValueError(f'The length of Co_vec must be {N}, but got {Co_vec.size}')
    X_vec = jnp.linspace(0.0, 1.0, N)

    phi_left, phi_right = -1.0, 1.0
    Co0, Co1 = Co_vec[0], Co_vec[-1]
    phio0, phio1 = phio_vec[0], phio_vec[-1]

    N_sample = 1000
    X_sample_vec = jnp.linspace(0.0, 1.0, N_sample)

    Co_vec_ = jnp.interp(X_sample_vec, X_vec, Co_vec)
    phio_vec_ = jnp.interp(X_sample_vec, X_vec, phio_vec)

    C1_vec = 0.5*Co_vec_ + innersol.Cpmi_left(Co0, phio0, epsilon, -1) - 0.5*Co0 + innersol.Cpmi_right(Co1, phio1, epsilon, -1) - 0.5*Co1
    C2_vec = 0.5*Co_vec_ + innersol.Cpmi_left(Co0, phio0, epsilon, 1) - 0.5*Co0 + innersol.Cpmi_right(Co1, phio1, epsilon, 1) - 0.5*Co1
    phi_vec = phio_vec_ + innersol.phii_left(Co0, phio0, epsilon) - phio0 + innersol.phii_right(Co1, phio1, epsilon) - phio1

    return C1_vec, C2_vec, phi_vec
