from pathlib import Path

import pysindy as ps
from exp.expdata import ExpData, LorenzExp
from exp.metrics import coeff_metrics, data_metrics
import jax.numpy as jnp
import jax.numpy as jnp
import numpy as np
import pickle
import time

def noise_dt_exp(noise_var, dt:0.05, save_path:str = None, exp_data: ExpData = LorenzExp):
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
    base_opt = ps.STLSQ(threshold=0.25, alpha=0.)
    optimizer = ps.optimizers.EnsembleOptimizer(
        opt = base_opt,
        library_ensemble = True,
        n_models = 2000,
        n_subset = int(0.8*expdata.t_train.shape[0]),
        replace=True,
        ensemble_aggregator = lambda x: np.mean(x,axis=0)
)


    model = ps.SINDy(
        feature_library=poly_lib, 
        optimizer=optimizer,
        feature_names=['x','y','z'],
        differentiation_method=ps.SmoothedFiniteDifference()
    )

    metrics = {}

    metrics["coeff_mets"]  = coeff_metrics(
        coeff_est = np.array(model.predict(x=expdata.x_true)),
        coeff_true = expdata.true_coeff
    )

    metrics["data_mets"] = data_metrics(
        pred_sim = model.predict(expdata.x_true),
        true = expdata.x_dot
    )

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f)

    return metrics



def big_experiment(exp_folder: str = "esindy_results"):
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
    big_experiment(exp_folder="esindy_results")