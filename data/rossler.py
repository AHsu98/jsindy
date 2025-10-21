import jax 
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import diffrax

def rossler_system(t,state,args):
    x,y,z = state
    a,b,c = args

    dxdt = -y-z
    dydt = x+a*y
    dzdt = b+z*(x-c)

    return jnp.array([dxdt,dydt, dzdt])

def solve_rossler(
        a=0.2,
        b=0.2,
        c=5.7,
        initial_state = jnp.array([-6,5,0]),
        t0=0.0,
        t1=20.0,
        dt=0.01
):
    args = (a,b,c)
    term = diffrax.ODETerm(rossler_system)
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
