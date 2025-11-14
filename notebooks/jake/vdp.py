# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: env
#     language: python
#     name: python3
# ---

# %%
from exp.expdata import GenExAdapter
import matplotlib.pyplot as plt
import numpy as np

from pysindy import STLSQ

from exp.evaluate.jsindy import evaluate_jmodel
from jsindy.sindy_model import JSINDyModel
from jsindy.trajectory_model import CholDataAdaptedRKHSInterpolant, CholRKHSInterpolant
from jsindy.dynamics_model import FeatureLinearModel, PolyLib
from jsindy.optim import AlternatingActiveSetLMSolver, LMSettings, AnnealedAlternatingActiveSetLMSolver
from jsindy.optim.solvers.alt_active_set_lm_solver import pySindySparsifier
from jsindy.kernels import ConstantKernel, ScalarMaternKernel


# %% [markdown]
# In this experiment, we know we can get correct answer with t1=100 and n_train=50 and the ActiveSetOptimizer
# Let's try to find how well we can do as we decrease the amount of training data, modifying
#
# `noise`, `t`, `n_train`, and whether we're using annealing

# %%
noise = 0.1
t1 = 100
n_train = 50
annealing = False

# %% [markdown]
# ## Generate Data

# %%
expdata = GenExAdapter(
    system="vdp",
    t0=0.,
    ic_std=0.02,
    t1=t1,
    noise=noise,
    n_train=n_train,
    n_colloc=500,
    seed=1234
)
x_true = expdata.x_true
t_true = expdata.t_true

t_train = expdata.t_train
x_train = expdata.x_train
t_train.shape
print(f"dt_train={t1/n_train}")
print(f"Noise std deviation is {100 * noise / np.std(x_true):.2f}% training data std deviation")
plt.plot(x_true[:,0], x_true[:,1], label="true system")
plt.xlabel(r"$x$")
plt.ylabel(r"$\dot x$")
plt.plot(x_train[:,0], x_train[:,1], "rx", label="Measurements")
plt.legend()
None

# %% [markdown]
# ## Define Model Parameters

# %%
kernel = (
    ConstantKernel(variance = 5.)
    +ScalarMaternKernel(p = 5,variance = 10., lengthscale=3,min_lengthscale=2.0)
)   

trajectory_model = CholDataAdaptedRKHSInterpolant(kernel=kernel)
dynamics_model = FeatureLinearModel(
    reg_scaling = 1.,
    feature_map=PolyLib(degree=3)
    
)
optsettings = LMSettings(
    max_iter = 2000,
    atol_gradnorm=1e-8,
    show_progress=True,
    no_tqdm=False,
    min_alpha = 1e-16,
    init_alpha = 5.,
)
sparsifier = pySindySparsifier(STLSQ(threshold = 0.25,alpha = 0.01))

if not annealing:
    optimizer = AlternatingActiveSetLMSolver(
        beta_reg=1e-3,
        solver_settings=optsettings,
        fixed_colloc_weight=1e5,
        fixed_data_weight=1,
        sparsifier = sparsifier,
    )
else:
    optimizer = AnnealedAlternatingActiveSetLMSolver(
        beta_reg=1e-3,
        solver_settings=optsettings,
        fixed_colloc_weight=1e5,
        fixed_data_weight=1,
        sparsifier = sparsifier,
        num_annealing_steps=4,
    )

model = JSINDyModel(
    trajectory_model=trajectory_model,
    dynamics_model=dynamics_model,
    optimizer=optimizer,
    input_orders=(0, 1),
    ode_order=2,
    feature_names=['x']
)

# %% [markdown]
# ## Fit the model

# %%
model.fit(expdata.t_train, expdata.x_train[:, :1], expdata.t_colloc)

print("True model:")
expdata.print()
print("Discovered model:")
model.print()

# %%
x_pred = model.predict_state(expdata.t_true)
xdot_pred = model.traj_model.derivative(expdata.t_true,model.z,diff_order = 1)


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6), sharex=True)

fontname = 'serif'

ax1.plot(t_true, x_true[:,0], label="true", c='black',)
ax1.plot(t_true, x_pred, label='jsindy prediction',linestyle='--',c='tab:red')
ax1.scatter(t_train, x_train[:,0], label="measurement", marker='d',s=80,c='black')
ax1.set_ylabel(r"$x$: Potential", fontname=fontname,size=12)
ax1.legend(prop={'family': fontname, 'size':12})
ax1.grid(True)

ax2.plot(t_true, x_true[:,1], label='predator', c='black')
ax2.plot(t_true, xdot_pred, label='prediction', linestyle='--', c='tab:red')
ax2.scatter(t_train, x_train[:,1], label="measurement", c='black', marker='d',s=80)
ax2.set_ylabel(r"$x'$", fontname=fontname,size=12)
ax2.set_xlabel("Time")
ax2.grid(True)
# ax2.legend(prop={'family': fontname, 'size':12})

plt.suptitle("Van der Pol Oscillator", fontsize=16, fontname=fontname)
plt.tight_layout()
plt.show()

