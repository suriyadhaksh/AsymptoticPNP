import jax.numpy as jnp
from jax import jit


@jit
def forwardTransform(epsilon, mathscrC0, mathscrC1, varrho0, varrho1, phi_left=-1.0, phi_right=1.0):
    """
    Transform boundary layer parameters to outer solution boundary conditions
    Co and phio.
    """

    Co0 = (
        mathscrC0**2
        * (mathscrC0**2 - varrho0**2)**2
        / (32.0 * epsilon**2 * varrho0**4)
    )

    Co1 = (
        mathscrC1**2
        * (mathscrC1**2 - varrho1**2)**2
        / (32.0 * epsilon**2 * varrho1**4)
    )

    T0 = -varrho0 / mathscrC0
    T1 = -varrho1 / mathscrC1

    phio0 = phi_left  - 4.0 * jnp.arctanh(T0)
    phio1 = phi_right - 4.0 * jnp.arctanh(T1)

    return Co0, Co1, phio0, phio1


@jit
def inverseTransform(epsilon, Co0, Co1, phio0, phio1, phi_left=-1.0, phi_right=1.0):
    """
    Transform outer solution boundary conditions Co and phio
    to boundary layer parameters mathscrC and varrho.
    """

    mathscrC0 = 4 * epsilon * jnp.sqrt(2 * Co0) * jnp.sinh((phi_left  - phio0) / 4)**2
    mathscrC1 = 4 * epsilon * jnp.sqrt(2 * Co1) * jnp.sinh((phi_right - phio1) / 4)**2
    varrho0   = -4 * epsilon * jnp.sqrt(2 * Co0) * jnp.tanh((phi_left  - phio0) / 4) * jnp.sinh((phi_left  - phio0) / 4)**2
    varrho1   = -4 * epsilon * jnp.sqrt(2 * Co1) * jnp.tanh((phi_right - phio1) / 4) * jnp.sinh((phi_right - phio1) / 4)**2

    return mathscrC0, mathscrC1, varrho0, varrho1