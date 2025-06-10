
import jax
from exp.expdata import ExpData
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


from jsindy import JSINDyModel
import numpy as np

def precision_score(coeff_est, coeff_true):
    est_bool = coeff_est != 0
    true_bool = coeff_true != 0

    true_positive = jnp.count_nonzero(est_bool[true_bool])
    false_positive = jnp.count_nonzero(est_bool[~true_bool])

    if (true_positive + false_positive) == 0:
        return jnp.nan
    
    return true_positive / (true_positive + false_positive)

def recall_score(coeff_est,coeff_true):
    est_bool = coeff_est != 0
    true_bool = coeff_true != 0

    true_positive = jnp.count_nonzero(est_bool[true_bool])
    false_negative = jnp.count_nonzero(~est_bool[true_bool])

    if (true_positive + false_negative) == 0:
        return jnp.nan

    return true_positive / (true_positive + false_negative)

def f1_score(est, true):
    est_bool = est != 0
    true_bool = true != 0

    true_positive = jnp.count_nonzero(est_bool[true_bool])
    false_positive = jnp.count_nonzero(est_bool[~true_bool])
    false_negative = jnp.count_nonzero(~est_bool[true_bool])

    denom = 2*true_positive + false_positive + false_negative

    if denom == 0:
        return jnp.nan
    
    return 2*true_positive / denom



def coeff_metrics(coeff_est, coeff_true) -> dict:
    est = coeff_est.flatten()
    true = coeff_true.flatten()
    metrics = {}
    metrics["precision"] = float(precision_score(est,true))
    metrics["recall"] = float(recall_score(est,true))
    metrics["f1"] = float(f1_score(est,true))

    metrics["coeff_rel_l2"]=float(jnp.linalg.norm(est-true)/jnp.linalg.norm((true)))

    metrics["coeff_rmse"] = float(jnp.sqrt(jnp.mean((est-true)**2)))
    metrics["coeff_mae"] = float(jnp.mean(jnp.abs(est-true)))

    return metrics


def data_metrics(pred_sim, true) -> dict:
    est = pred_sim.flatten()
    true = true.flatten()

    metrics = {}

    mse = np.mean((true - est) ** 2)
    mae = np.mean(np.abs(true - est))
    rmse = np.sqrt(mse)
    max_error = np.max(np.abs(true - est))

    var_true = np.var(true)
    norm_mse = mse / var_true if var_true > 0 else np.nan

    rel_l2_error = (
        np.linalg.norm(est - true) / 
        np.linalg.norm(true) if np.linalg.norm(true) > 
        0 else np.nan
    )


    metrics["mse"] = mse
    metrics["rmse"] = rmse
    metrics["mae"] = mae
    metrics["max_abs_error"] = max_error
    metrics["normalized_mse"] = norm_mse
    metrics["relative_l2_error"] = rel_l2_error

    return metrics



def get_model_metrics(model: JSINDyModel, exp_data: ExpData) -> tuple[JSINDyModel, dict]:  
    """
    Fit a JSINDyModel model on exp_data and return trained model with dictionary
    of metrics.
    """

    # fit model
    model.fit(
        x=exp_data.x_train,
        t=exp_data.t_train,
        t_colloc = exp_data.t_colloc
        )

    coeff = model.coef_
    true_coeff = exp_data.true_coeff

    metrics = coeff_metrics(coeff, true_coeff)

    # predict 
    x_pred = model.x_predict(exp_data.t_true)
    x_true = exp_data.x_true

    # interpolation error
    inter_err  = jnp.linalg.norm(x_pred - x_true) / jnp.linalg.norm(x_true)

    # extrapolation error 
    x_test_pred = model.simulate(
        x0=exp_data.x_test[0],
        t0=exp_data.t0,
        t1=exp_data.t1,
        dt_eval = exp_data.dt,
    )
    x_test = exp_data.x_test

    extrap_err = jnp.linalg.norm(x_test_pred-x_test) / jnp.linalg.norm(x_test)

    metrics["interp_err"] = inter_err
    metrics["extrap_err"] = extrap_err

    return model, metrics


def run_avg_err(true_arr, est_arr):
    n_samples = len(true_arr)

    err = []
    for i in range(n_samples):
        try:
            err_i = jnp.linalg.norm(true_arr[:i+1]-est_arr[:i+1]) / jnp.linalg.norm(true_arr[:i+1])
        except:
            err_i=jnp.nan
        err.append(err_i)
    return err

