import jax 
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import diffrax

def lotka_volterra_system(t, state, args):
    x, y = state
    alpha, beta, gamma, delta = args

    dxdt = alpha*x - beta*(x*y)
    dydt = - gamma*y + delta*(x*y)

    return jnp.array([dxdt,dydt])

def solve_lotka_voltera(
        alpha=1.1,
        beta=0.4,
        gamma=0.4,
        delta=0.1,
        initial_state=jnp.array([10.,5.]),
        t0=0.0,
        t1=20.0,
        dt=0.01,
):
    args =(alpha, beta, gamma, delta)
    term = diffrax.ODETerm(lotka_volterra_system)
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
        max_steps=int(10*(t1-t0)/dt)
    )

    return sol