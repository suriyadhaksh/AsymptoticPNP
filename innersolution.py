import jax
import jax.numpy as jnp
from functools import partial

'''
(Differentiable) Analytical solutions for the inner region of the Perturbed 1D PNP Blocking electrode problem.
Used to construct the projection (P) operator. See [../docs/api/innersolution.md] for details.
'''

@partial(jax.jit, static_argnames=('epsilon', 'X_vec', 'phi_left'))
def phii_left(Co0, phio0, epsilon, X_vec, phi_left=-1.0):
    x_vec = X_vec / epsilon

    K1 = jnp.tanh(0.25 * (phi_left - phio0))
    K2 = -jnp.sqrt(Co0 / 2.0)
    return phio0 + 4 * jnp.arctanh(K1 * jnp.exp(K2 * x_vec))

    # TODO: Add check for numerical stability when epsilon → 0
    arg = jnp.clip(K1 * jnp.exp(K2 * x_vec), -0.999999, 0.999999)
    return phio0 + 4 * jnp.arctanh(arg)


@partial(jax.jit, static_argnames=('epsilon', 'X_vec', 'phi_right'))
def phii_right(Co1, phio1, epsilon, X_vec, phi_right=1.0):
    X_right = X_vec[-1]
    x_vec = (X_right - X_vec) / epsilon

    K1 = jnp.tanh(0.25 * (phi_right - phio1))
    K2 = -jnp.sqrt(Co1 / 2.0)
    return phio1 + 4 * jnp.arctanh(K1 * jnp.exp(K2 * x_vec))

    # TODO: Add check for numerical stability when epsilon → 0
    arg = jnp.clip(K1 * jnp.exp(K2 * x_vec), -0.999999, 0.999999)
    return phio0 + 4 * jnp.arctanh(arg)

@partial(jax.jit, static_argnames=('epsilon', 'X_vec', 'phi_left'))
def Ci_left(Co0, phio0, epsilon, X_vec, phi_left=-1.0):
    phii = phii_left(Co0, phio0, epsilon, X_vec, phi_left)
    return Co0 * jnp.cosh(phii - phio0)

@partial(jax.jit, static_argnames=('epsilon', 'X_vec', 'phi_right'))
def Ci_right(Co1, phio1, epsilon, X_vec, phi_right=1.0):
    phii = phii_right(Co1, phio1, epsilon, X_vec, phi_right)
    return Co1 * jnp.cosh(phii - phio1)

@partial(jax.jit, static_argnames=('plusOrMinus', 'epsilon', 'X_vec', 'phi_left'))
def Cpmi_left(Co0, phio0, epsilon, plusOrMinus, X_vec, phi_left=-1.0):
    # Use plusOrMinus = -1 for C+ and +1 for C-
    phii = phii_left(Co0, phio0, epsilon, X_vec, phi_left)
    return 0.5 * Co0 * jnp.exp(plusOrMinus * (phii - phio0))

@partial(jax.jit, static_argnames=('plusOrMinus', 'epsilon', 'X_vec', 'phi_left'))
def Cpmi_right(Co1, phio1, epsilon, plusOrMinus, X_vec, phi_right=1.0):
    # Use plusOrMinus = -1 for C+ and +1 for C-
    phii = phii_right(Co1, phio1, epsilon, X_vec, phi_right)
    return 0.5 * Co1 * jnp.exp(plusOrMinus * (phii - phio1))