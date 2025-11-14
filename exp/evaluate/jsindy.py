import jax
jax.config.update('jax_enable_x64',True)

from pathlib import Path

from jsindy.sindy_model import JSINDyModel
from jsindy.trajectory_model import DataAdaptedRKHSInterpolant
from jsindy.dynamics_model import FeatureLinearModel
from jsindy.optim import AlternatingActiveSetLMSolver, LMSettings

from exp.expdata import ExpData, GenExAdapter
from exp.metrics import coeff_metrics, data_metrics
import pickle
import jax.numpy as jnp
from gen_experiments.utils import coeff_metrics as legacy_coeff_metrics
from gen_experiments.utils import pred_metrics, unionize_coeff_matrices
import pysindy as ps

import time


def evaluate_jmodel(model:JSINDyModel, expdata:ExpData,  save_path: str = None):

    model.fit(
        expdata.t_train,
        expdata.x_train,
        expdata.t_colloc
    )

    metrics = {}
    
    metrics["theta"] = model.theta

    # states prediction 
    metrics["x_metrics"] = data_metrics(
        pred_sim = model.predict_state(expdata.t_true),
        true = expdata.x_true
    )

    # save learned coeffs
    metrics["theta_metrics"]  = coeff_metrics(
        coeff_est = model.theta.T,
        coeff_true = expdata.true_coeff
    )

    metrics["xdot_metrics"] = data_metrics(
        pred_sim = model.predict(expdata.x_true),
        true = expdata.x_dot
    )

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f)
    
    return metrics


def evaluate_jmodel_alt(model:JSINDyModel, expdata:GenExAdapter,  save_path: str = None):

    # proxy_model for compatability with regular sindy functions
    # Definitely not type safe, can only use proxy_model with limited functions
    proxy_model = ps.SINDy(feature_library = model.dynamics_model.feature_map)
    proxy_model =  ps.SINDy(feature_library = model.dynamics_model.feature_map)
    proxy_model.model = True
    proxy_model.feature_names = expdata.feature_names
    proxy_model.coefficients = lambda: model.theta.T

    coeff_true, coeff_est, names = unionize_coeff_matrices(proxy_model, expdata.true_coeff)
    metrics = {}
    
    # metrics["theta"] = model.theta

    # states prediction 
    metrics["x_metrics"] = data_metrics(
        pred_sim = model.predict_state(expdata.t_true),
        true = expdata.x_true
    )

    # save learned coeffs
    metrics["theta_metrics"]  = legacy_coeff_metrics(coeff_est, coeff_true)

    metrics["xdot_metrics"] = data_metrics(
        pred_sim = model.predict(expdata.x_true),
        true = expdata.x_dot
    )

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f)
    
    return metrics

