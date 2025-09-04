import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import jax
jax.config.update('jax_enable_x64',True)
from pathlib import Path

from jsindy.sindy_model import JSINDyModel
from jsindy.trajectory_model import CholDataAdaptedRKHSInterpolant
from jsindy.dynamics_model import FeatureLinearModel, PolyLib
from jsindy.optim import AlternatingActiveSetLMSolver, LMSettings
from jsindy.optim.solvers.alt_active_set_lm_solver import pySindySparsifier
from pysindy import STLSQ
from exp.expdata import LinearExp
from exp.evaluate.jsindy import evaluate_jmodel
import pickle
import jax.numpy as jnp

import time

def create_linear_experiment_data(noise, dt):
    initial_state = jnp.array([1., 0., 0., 1., -1., 0.])
    # sigma^2 - var
    true_sigma2 = noise
    t0=0
    t1=10.1
    n_train = len(jnp.arange(t0,t1,dt))

    n_colloc = 500
    expdata = LinearExp(
        initial_state=initial_state,
        t0=t0,
        t1=t1,
        dt = 0.01,
        dt_train=dt,
        noise= jnp.sqrt(true_sigma2),
        seed=29,
        n_colloc=n_colloc,
        one_rkey=True,
        # feature_names=['x','y','z']
    )
    return expdata

def create_jsindy_model(feature_names = None):
    trajectory_model = CholDataAdaptedRKHSInterpolant()
    dynamics_model = FeatureLinearModel(
        reg_scaling = 1.,
        feature_map=PolyLib(degree=1)
    )
    optsettings = LMSettings(
        max_iter = 2000,
        show_progress=False,
        no_tqdm=True,
        min_alpha = 1e-16,
        init_alpha = 5.,
    )
    data_weight =  1.
    colloc_weight = 50.
    sparsifier = pySindySparsifier(STLSQ(threshold = 0.1,alpha = 1.))
    optimizer = AlternatingActiveSetLMSolver(
            beta_reg=0.001,
            solver_settings=optsettings,
            fixed_colloc_weight=colloc_weight,
            fixed_data_weight=data_weight,
            sparsifier = sparsifier
            )
    
    model = JSINDyModel(
        trajectory_model=trajectory_model,
        dynamics_model=dynamics_model,
        optimizer=optimizer,
        feature_names=feature_names
    )
    return model


def noise_dt_experiment(exp_folder: str = "jsindy_linear_results"):
    folder = Path(exp_folder)
    folder.mkdir(parents=True, exist_ok=True)

    noise_vars = jnp.array([0,0.01,0.05,0.1])[::-1]
    dt_vals = jnp.around(jnp.array([0.05, 0.1, 0.2,0.5]),4)[::-1]

    noise_vars = [jnp.array([0,0.01,0.05,0.1])[::-1][-1]]
    dt_vals = [jnp.around(jnp.array([0.05, 0.1, 0.2,0.5]),4)[::-1][1]]


    tot_exp = len(noise_vars)*len(dt_vals)

    idx=1
    for noise in noise_vars:
        for dt in dt_vals:
            print(f"Starting dt = {float(jnp.around(dt,3))}, noise = {float(jnp.around(noise,3))}")
            start_time = time.time()
            exp_name = f"noise_{float(jnp.around(noise,3))}_dt_{float(jnp.around(dt,3))}.dill"
            save_path = folder / exp_name

            try: 
                expdata = create_linear_experiment_data(noise,dt)
                jmodel = create_jsindy_model(expdata.feature_names)
                metrics = evaluate_jmodel(jmodel,expdata)
            except Exception as e:
                print(f"Error: {e}")
                continue

            with open(save_path, 'wb') as f:
                pickle.dump(metrics,f)

            l2_err = metrics["xdot_metrics"]["relative_l2_error"]

            tot_time = time.time() - start_time
            
            print(
                f"Finished exp {idx}/{tot_exp}, "
                f"noise: {float(jnp.around(noise,3))}, dt: {float(jnp.around(dt,3))}, "
                f"xdot err = {l2_err:.4e}, time = {tot_time:3.2f}."
            )
            idx += 1

if __name__ == "__main__":
    noise_dt_experiment(exp_folder="jsindy_linear_results/june27/results")