import jax.numpy as jnp

'''
(Differentiable) Analytical solutions for the inner region of the Perturbed 1D PNP Blocking electrode problem.
Used to construct the projection (P) operator. See [../docs/api/innersolution.md] for details.
'''

def phii_left(Co0, phio0, epsilon):
    N = 1000
    X_vec = jnp.linspace(0.0, 1.0, N)
    x_vec = X_vec / epsilon

    phi_left = -1.0

    K1 = jnp.tanh(0.25 * (phi_left - phio0))
    K2 = -jnp.sqrt(Co0 / 2.0)
    return phio0 + 4 * jnp.arctanh(K1 * jnp.exp(K2 * x_vec))

def phii_right(Co1, phio1, epsilon):
    N = 1000
    X_vec = jnp.linspace(0.0, 1.0, N)
    x_vec = (1.0 - X_vec) / epsilon

    phi_right = 1.0

    K1 = jnp.tanh(0.25 * (phi_right - phio1))
    K2 = -jnp.sqrt(Co1 / 2.0)
    return phio1 + 4 * jnp.arctanh(K1 * jnp.exp(K2 * x_vec))

def Ci_left(Co0, phio0, epsilon):
    phii = phii_left(Co0, phio0, epsilon)
    return Co0 * jnp.cosh(phii - phio0)

def Ci_right(Co1, phio1, epsilon):
    phii = phii_right(Co1, phio1, epsilon)
    return Co1 * jnp.cosh(phii - phio1)

def Cpmi_left(Co0, phio0, epsilon, plusOrMinus):
    # Use plusOrMinus = -1 for C+ and +1 for C-
    phii = phii_left(Co0, phio0, epsilon)
    return 0.5 * Co0 * jnp.exp(plusOrMinus * (phii - phio0))

def Cpmi_right(Co1, phio1, epsilon, plusOrMinus):
    # Use plusOrMinus = -1 for C+ and +1 for C-
    phii = phii_right(Co1, phio1, epsilon)
    return 0.5 * Co1 * jnp.exp(plusOrMinus * (phii - phio1))