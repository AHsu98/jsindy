from pathlib import Path

import pysindy as ps
from exp.expdata import ExpData, LorenzExp
from exp.metrics import coeff_metrics, data_metrics
import jax.numpy as jnp
import jax.numpy as jnp
import numpy as np
import pickle
import time

def noise_dt_exp(noise_var, dt, save_path: str = None, exp_data: ExpData = LorenzExp):

    initial_state = jnp.array([ 0.37719066, -0.39631459, 16.92126795])
    true_sigma2 = noise_var
    n_colloc = None
    expdata = exp_data(
        initial_state=initial_state,
        t0=0.0,
        t1=10.01,
        dt = 0.01,
        dt_train=dt,
        noise= jnp.sqrt(true_sigma2),
        seed=32,
        n_colloc=n_colloc,
        one_rkey=True,
        feature_names=['x','y','z']
    )

    poly_lib = ps.PolynomialLibrary(degree=2)

    ode_lib = ps.WeakPDELibrary(
        function_library=poly_lib,  
        spatiotemporal_grid=expdata.t_train,
        differentiation_method=ps.SmoothedFiniteDifference,
        K=1000,
    )

    # Instantiate and fit the SINDy model with the integral of u_dot
    optimizer = ps.SR3(
        reg_weight_lam=0.1,
        regularizer="L0",
        max_iter=1000,
        normalize_columns=True,
        tol=1e-1
    )


    model = ps.SINDy(
        feature_library=ode_lib, 
        optimizer=optimizer,
        feature_names=['x','y','z']
    )
    model.fit(expdata.x_train)

    metrics = {}

    metrics["coeff_mets"]  = coeff_metrics(
        coeff_est = model.coefficients(),
        coeff_true = expdata.true_coeff
    )

    metrics["data_mets"] = data_metrics(
        pred_sim = np.array(model.predict(x=expdata.x_true)),
        true = expdata.x_dot
    )

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(metrics,f)

    return metrics




def big_experiment(exp_folder: str = "wsindy_results"):
    folder = Path(exp_folder)
    folder.mkdir(parents=True,exist_ok=True)

    noise_vars = jnp.linspace(0,20,11)
    dt_vals = jnp.linspace(0,0.2,11)[1:]

    tot_exp = len(noise_vars)*len(dt_vals)
    idx=1
    for noise in noise_vars:
        for dt in dt_vals:
            start_time = time.time()
            exp_name = f"noise_{float(noise)}_dt_{float(dt)}.dill"
            save_path = folder / exp_name

            mets = noise_dt_exp(noise,dt,save_path)
            l2_err = mets["data_mets"]["relative_l2_error"]

            tot_time = time.time() - start_time
            
            print(
                f"exp {idx}/{tot_exp}, "
                f"noise: {float(noise)}, dt: {float(dt)}, "
                f"err = {l2_err}, time = {tot_time}."
            )
            idx += 1


if __name__ == "__main__":
    big_experiment(exp_folder="wsindy_results")