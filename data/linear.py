import os
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import diffrax

from jax.scipy.linalg import block_diag

@jax.jit
def linear_system(t,y,args):
    A1, A2, A3 = args
    A = block_diag(A1,A2,A3)

    return A @ y

def solve_linear(
        A1=None, 
        A2=None, 
        A3=None,
        initial_state=jnp.array([1., 0., 0., 1., -1., 0.]),
        t0=0.0,
        t1=20.0,
        dt=0.01
):
    if A1 is None:
        A1 = jnp.array([[0, 1],
                        [-2, -3]])
    if A2 is None:
        A2 = jnp.array([[0, -1],
                        [4,  0]])
    if A3 is None:
        A3 = jnp.array([[-1, 2],
                        [-2, -1]])
        
    args = (A1,A2,A3)
    term = diffrax.ODETerm(linear_system)
    solver = diffrax.Tsit5()

    save_at = diffrax.SaveAt(dense=True)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=initial_state,
        args=args,
        saveat=save_at,
        max_steps = int(10*(t1-t0)/dt)
    )
    return sol


