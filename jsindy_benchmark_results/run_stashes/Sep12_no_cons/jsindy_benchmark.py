import jax
print(jax.devices())
jax.config.update('jax_default_device',jax.devices()[5])
jax.config.update('jax_enable_x64',True)

from jax.random import key
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm
from exp.expdata import LorenzExp
import jax.numpy as jnp
import matplotlib.pyplot as plt
from exp.metrics import coeff_metrics, data_metrics
plt.style.use("ggplot")

from jsindy.sindy_model import JSINDyModel
from jsindy.util import get_collocation_points_weights
from jsindy.trajectory_model import DataAdaptedRKHSInterpolant,CholDataAdaptedRKHSInterpolant
from jsindy.dynamics_model import FeatureLinearModel, PolyLib
from jsindy.optim import AlternatingActiveSetLMSolver, LMSettings
from jsindy.optim.solvers.alt_active_set_lm_solver import pySindySparsifier
from pysindy import STLSQ,SSR,MIOSR
from jsindy.kernels import ConstantKernel, ScalarMaternKernel
import pickle
from pathlib import Path

x0 = jnp.array([-8, 8, 27.])
dt = 0.01
t0=0
t1=10.1
n_colloc = 505

expdata = LorenzExp(
    dt = dt,
    initial_state=x0,
    feature_names=['x','y','z'],
    t0=t0,
    t1=t1,
    n_colloc=n_colloc
)

tEndL = jnp.arange(4.0, 11.0, 1.0)
epsL = jnp.arange(0.025, 0.401, 0.025)

t_true = expdata.t_true
X_true = expdata.x_true

cutoff = 1
signal_power = jnp.std(X_true)
n_colloc = 500

def noise_time_exp(noise_ratio,tend,rkey = 0, save_path=None):
    print("------------------------------------------------------------------------")
    print(f"NOISE RATIO: {noise_ratio}")
    print(f"T end: {tend}")
    t_end_idx = int(tend // dt)
    X_train = X_true[:t_end_idx]
    t_train = t_true[:t_end_idx]

    t_colloc, w_colloc = get_collocation_points_weights(t_train,n_colloc)

    eps = noise_ratio*signal_power


    noise = eps*jax.random.normal(rkey, X_train.shape)

    X_train = X_train + noise

    kernel = (
        ConstantKernel(variance = 5.)
        +ScalarMaternKernel(p = 5,variance = 10., lengthscale=3,min_lengthscale=0.05)
    )   
    trajectory_model = CholDataAdaptedRKHSInterpolant(kernel=kernel)
    dynamics_model = FeatureLinearModel(
        reg_scaling = 1.,
        feature_map=PolyLib(degree=2,include_bias = False)
    )
    optsettings = LMSettings(
        max_iter = 1000,
        no_tqdm=True,
        min_alpha = 1e-16,
        init_alpha = 5.,
        print_every = 100,
        show_progress = True,
    )
    data_weight =  1.
    colloc_weight = 1e5
    sparsifier = pySindySparsifier(
        STLSQ(threshold = 0.2,alpha = 0.01)
        )


    optimizer = AlternatingActiveSetLMSolver(
            beta_reg=1e-3,
            solver_settings=optsettings,
            fixed_colloc_weight=colloc_weight,
            fixed_data_weight=data_weight,
            sparsifier = sparsifier
            )

    model = JSINDyModel(
        trajectory_model=trajectory_model,
        dynamics_model=dynamics_model,
        optimizer=optimizer,
        feature_names=expdata.feature_names
    )

    model.fit(t_train, X_train,t_colloc=t_colloc,w_colloc = w_colloc)

    metrics = {}

    metrics["coeff_mets"] = coeff_metrics(
        coeff_est=model.theta,
        coeff_true=expdata.true_coeff.T[1:]
    )
    metrics["theta"] = model.theta
    metrics['noise_ratio'] = noise_ratio
    metrics['t_end'] = tend

    if save_path: 
        with open(save_path, 'wb') as file:
            pickle.dump(metrics,file)
    model.print()
    print(model)
    print(metrics['coeff_mets']["coeff_rel_l2"])

    return metrics

if __name__ == "__main__":

    folder = "results"

    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    main_key = key(1)
    num_repeats = 8
    all_keys = jax.random.split(main_key,num_repeats)
    for i,rkey in enumerate(all_keys):
        for tend in tqdm(tEndL):
            for noise_ratio in epsL:
                exp_name = f"run_{i}_noise_ratio_{float(jnp.around(noise_ratio,3))}_t_end_{float(jnp.around(tend,3))}.pkl"
                save_path = folder / exp_name
                mets = noise_time_exp(noise_ratio, tend, rkey,save_path=save_path)


