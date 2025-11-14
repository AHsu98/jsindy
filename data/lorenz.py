import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import diffrax

def lorenz_system(t, state, args):
    x, y, z = state
    sigma, rho, beta = args
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return jnp.array([dxdt, dydt, dzdt])

def solve_lorenz(sigma=10.0, rho=28.0, beta=8.0/3.0, 
                 initial_state=jnp.array([1., 1.,1.]), 
                 t0=0.0, t1=20.0, dt=0.001):
    """Solves the Lorenz system.

    Args:
        sigma, rho, beta: Parameters of the Lorenz system.
        initial_state: Initial conditions [x0, y0, z0].
        t0, t1: Start and end time.
        dt: Time step size.

    Returns:
        ts: Array of time values.
        ys: Array of state values over time.
    """
    args = (sigma, rho, beta)
    term = diffrax.ODETerm(lorenz_system)
    stepsize_controller = diffrax.PIDController(atol = 1e-9,rtol = 1e-9)
    
    solver = diffrax.Tsit5()
    
    save_at = diffrax.SaveAt(dense=True)  # Save at regular intervals
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,  # Initial step size
        stepsize_controller=stepsize_controller,
        y0=initial_state,
        args=args,
        saveat=save_at,
        max_steps = int(100*(t1-t0)/dt)
    )
    
    return sol