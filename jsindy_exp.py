import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import jax
jax.config.update('jax_enable_x64',True)
from pathlib import Path

from jsindy.sindy_model import JSINDyModel
from jsindy.trajectory_model import DataAdaptedRKHSInterpolant
from jsindy.dynamics_model import FeatureLinearModel
from jsindy.optim import AlternatingActiveSetLMSolver, LMSettings

from exp.expdata import ExpData, LorenzExp
from exp.metrics import coeff_metrics, data_metrics
import pickle
import jax.numpy as jnp

import time

def noise_dt_exp(noise_var, dt, save_path:str = None, exp_data: ExpData = LorenzExp):
    """
    Get theta and prediction metrics for a selected variance and dt value
    """
    # generate experiment data
    initial_state = jnp.array([ 0.37719066, -0.39631459, 16.92126795])
    # sigma^2 - var
    true_sigma2 = noise_var
    t0=0
    t1=10.1
    n_train = len(jnp.arange(t0,t1,dt))

    n_colloc = 500
    expdata = exp_data(
        initial_state=initial_state,
        t0=t0,
        t1=t1,
        dt = 0.01,
        dt_train=dt,
        noise= jnp.sqrt(true_sigma2),
        seed=32,
        n_colloc=n_colloc,
        one_rkey=True,
        feature_names=['x','y','z']
    )

    trajectory_model = DataAdaptedRKHSInterpolant()
    dynamics_model = FeatureLinearModel(reg_scaling = 1.)
    optsettings = LMSettings(
        max_iter = 2000,
        show_progress=False,
        no_tqdm=True,
        min_alpha = 1e-16,
        init_alpha = 5.,
    )
    optimizer = AlternatingActiveSetLMSolver(
            beta_reg=0.001,
            solver_settings=optsettings,
            fixed_colloc_weight=50.)
    
    model = JSINDyModel(
        trajectory_model=trajectory_model,
        dynamics_model=dynamics_model,
        optimizer=optimizer,
        feature_names=expdata.feature_names
    )
    model.fit(
        expdata.t_train,
        expdata.x_train,
        expdata.t_colloc
    )

    metrics = {}

    metrics["coeff_mets"]  = coeff_metrics(
        coeff_est = model.theta.T,
        coeff_true = expdata.true_coeff
    )

    metrics["data_mets"] = data_metrics(
        pred_sim = model.predict(expdata.x_true),
        true = expdata.x_dot
    )
    metrics['model_params'] = model.params
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f)
    
    return metrics


def big_experiment(exp_folder: str = "jsindy_results"):
    folder = Path(exp_folder)
    folder.mkdir(parents=True,exist_ok=True)

    noise_vars = jnp.linspace(0,20,11)
    dt_vals = jnp.around(jnp.linspace(0,0.2,11)[1:],4)

    tot_exp = len(noise_vars)*len(dt_vals)
    idx=1
    for noise in noise_vars:
        for dt in dt_vals:
            print(f"Starting dt = {float(dt)}, noise = {float(noise)}")
            start_time = time.time()
            exp_name = f"noise_{float(noise)}_dt_{float(dt)}.dill"
            save_path = folder / exp_name

            mets = noise_dt_exp(noise,dt,save_path)
            l2_err = mets["data_mets"]["relative_l2_error"]

            tot_time = time.time() - start_time
            
            print(
                f"Finished exp {idx}/{tot_exp}, "
                f"noise: {float(jnp.around(noise,3))}, dt: {float(jnp.around(dt,3))}, "
                f"err = {l2_err:.4e}, time = {tot_time:3.2f}."
            )
            idx += 1


if __name__ == "__main__":
    big_experiment(exp_folder="jsindy_results/june18_test_chol/exp_results")