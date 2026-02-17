import jax
print(jax.devices())
jax.config.update('jax_default_device',jax.devices()[1])
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
import pysindy as ps
import time
import numpy as np
import warnings

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


true_theta = expdata.true_coeff

def get_dataset(rkey,t_end,noise_ratio):
    t_end_idx = int(t_end // dt)
    X_train = X_true[:t_end_idx]
    t_train = t_true[:t_end_idx]
    eps = noise_ratio*signal_power
    noise = eps*jax.random.normal(rkey, X_train.shape)
    X_train = X_train + noise
    return t_train,X_train

def fit_weak_sindy(t_train,X_train):
    library = ps.PolynomialLibrary(2)
    ode_lib = ps.WeakPDELibrary(
        function_library=library,
        spatiotemporal_grid=t_train,
        # is_uniform=True,
        K=250,
    )
    optimizer = ps.EnsembleOptimizer(
        STLSQ(threshold = 0.5,alpha = 0.05),
        bagging=True,
        n_models = 50
        )
    weak_model = ps.SINDy(feature_library=ode_lib, optimizer=optimizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weak_model.fit(np.array(X_train),np.array(t_train))
    return weak_model

def evaluate_coeffs(theta,theta_true):
    support_correct = jnp.all((theta==0)==(theta_true==0))
    coeff_error = jnp.linalg.norm(theta - theta_true)/jnp.linalg.norm(theta_true)
    return support_correct,coeff_error

tEndL = jnp.arange(4.0, 11.0, 1.0)
epsL = jnp.arange(0.025, 0.401, 0.025)
num_repeats = 128
base_key = key(4280)
all_keys = jax.random.split(base_key,num_repeats)
all_support = []
all_error = []
for rep in tqdm(range(num_repeats)):
    print(rep)
    support = np.zeros((len(tEndL),len(epsL)))
    error = np.zeros((len(tEndL),len(epsL)))
    for (i,t_end) in enumerate(tEndL):
        for (j,eps) in enumerate(epsL):
            t_train,X_train = get_dataset(all_keys[rep],t_end=t_end,noise_ratio = eps)
            weak_model = fit_weak_sindy(t_train,X_train)
            theta = weak_model.coefficients()
            recovered,coeff_error = evaluate_coeffs(theta,true_theta)
            support[i,j] = recovered
            error[i,j] = coeff_error
    all_support.append(support)
    all_error.append(error)

all_support = jnp.array(all_support)
all_error = jnp.array(all_error)

jnp.save("weak_ensemble_support.npy",all_support)
jnp.save("weak_ensemble_error.npy",all_error)