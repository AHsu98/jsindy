import jax
jax.config.update('jax_enable_x64',True)
jax.config.update("jax_default_device",jax.devices()[1])
import jax.numpy as jnp
import diffrax

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from jax.random import key

from jsindy.sindy_model import JSINDyModel
from jsindy.trajectory_model import CholDataAdaptedRKHSInterpolant
from jsindy.dynamics_model import FeatureLinearModel,PolyLib
from jsindy.optim import AlternatingActiveSetLMSolver, LMSettings,AnnealedAlternatingActiveSetLMSolver
from jsindy.optim.solvers.alt_active_set_lm_solver import pySindySparsifier,NoSparsifier
from pysindy import STLSQ,SR3,MIOSR,EnsembleOptimizer
import jsindy
from numpy import loadtxt

t1 = 10
x0 = jnp.array([2,0.])
def f(t,x,args):
    return jnp.array([- 0.1*x[0]**3 + 2*x[1]**3,- 0.1*x[1]**3 -2*x[0]**3])
ode = diffrax.ODETerm(f)
step_control = diffrax.PIDController(rtol = 1e-7,atol =1e-7)
solver = diffrax.Tsit5()
save_at = diffrax.SaveAt(dense = True)

sol = diffrax.diffeqsolve(ode,solver,0.,20.,dt0 = 0.005,y0 = x0,saveat = save_at,stepsize_controller=step_control)
t_grid = jnp.linspace(0,t1,500)
x_vals = jax.vmap(sol.evaluate)(t_grid)

dt_train = 0.05

t_train = jnp.arange(0,t1,dt_train)
x_train_true = jax.vmap(sol.evaluate)(t_train)


trajectory_model = CholDataAdaptedRKHSInterpolant()
dynamics_model = FeatureLinearModel(reg_scaling = 1.,feature_map=PolyLib(3))
optsettings = LMSettings(
    max_iter = 1000,
    show_progress=False,
    no_tqdm=False,
    min_alpha = 1e-16,
    init_alpha = 5.,
    print_every = 100,
)
data_weight =  10.
colloc_weight = 1e5
sparsifier = pySindySparsifier(
    EnsembleOptimizer(
    STLSQ(threshold = 0.05,alpha = 10.),
    bagging=True,
    n_models = 200)
    )

# sparsifier = NoSparsifier()

optimizer = AlternatingActiveSetLMSolver(
        beta_reg=1e-5,
        solver_settings=optsettings,
        fixed_colloc_weight=colloc_weight,
        fixed_data_weight=data_weight,
        sparsifier = sparsifier
        )

model = JSINDyModel(
    trajectory_model=trajectory_model,
    dynamics_model=dynamics_model,
    optimizer=optimizer,
    feature_names=['x','y']
)

true_theta = jnp.array(
    [[ 0.        ,  0.        ],
    [ 0.        , 0.    ],
    [0. ,  0.        ],
    [ 0.        ,  0.        ],
    [ 0.        ,  0.        ],
    [ 0.        ,  0.        ],
    [-0.1, -2.],
    [ 0.        ,  0.        ],
    [ 0.        ,  0.        ],
    [ 2., -0.1]])

all_xpreds = []
all_xdot_preds = []
all_thetas = []


num_repeats = 32
noise_vals = jnp.array([0.02,0.04,0.08,0.16,0.32,0.64])

all_theta_errors = jnp.zeros((num_repeats,len(noise_vals)))
all_x_errors = jnp.zeros((num_repeats,len(noise_vals)))
all_keys = jax.random.split(key(3184),num_repeats)


for repetition in range(num_repeats):
    print(repetition)
    for s,noise_sigma in enumerate(noise_vals):
        x_train = x_train_true + noise_sigma * jax.random.normal(all_keys[repetition],(x_train_true.shape))
        model.fit(t_train,x_train)
        model.print()

        xpred = model.predict_state(t_grid)
        all_theta_errors = all_theta_errors.at[repetition,s].set(
            jnp.linalg.norm(true_theta - model.theta)/jnp.linalg.norm(true_theta))
        
        all_x_errors = all_x_errors.at[repetition,s].set(
            jnp.sqrt(jnp.mean((x_vals - xpred)**2))/jnp.std(x_vals))
    
jnp.save('exp2/all_theta_errors.npy', all_theta_errors)
jnp.save('exp2/all_x_errors.npy', all_x_errors)