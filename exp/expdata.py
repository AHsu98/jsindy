import math
from typing import cast

import jax
from jax.random import PRNGKey
from data.linear import solve_linear, linear_system
from data.lorenz import solve_lorenz, lorenz_system
from data.lotkavolterra import solve_lotka_voltera, lotka_volterra_system
from data.rossler import solve_rossler, rossler_system
from jsindy.util import get_collocation_points_weights
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


import pysindy as ps
import gen_experiments.data as legacy_data

from dataclasses import dataclass,field
from typing import Optional
import numpy as np
import warnings

@dataclass
class ExpData:
    """
    t0 :
        starting time
    t1 : 
        end time
    dt : 
        time interval for true data 
    dt_train : 
        time interval used to generate training data 
    seed :
        seed value to create jax random keys for training and test data.
    noise :
        amount of noise (std) to add to training data. 
    ic_std :
        random noise (standard deviation) to add to initial condition (ic)
        to create test data. 
    n_colloc : 
        number of collocation points used in training. Default is the number of 
        training points.
    
    
    """
    t0: float = 0.0
    t1: float = 10.0
    dt: float = 0.01
    dt_train: float = 0.1
    ode_dt0: float = 0.001
    seed: int = 1234
    noise: float = 0.0
    ic_std: float = 2.
    n_colloc: Optional[int] = None
    one_rkey: bool = False

    def __post_init__(self):
        self.random_key = PRNGKey(self.seed)
        if self.one_rkey:
            self.train_key = self.random_key
            self.test_key = self.random_key
        else:
            self.train_key, self.test_key = jax.random.split(self.random_key,2)

        self.t_true = jnp.arange(self.t0,self.t1,self.dt)
        self.t_train = jnp.arange(self.t0,self.t1,self.dt_train)

        if self.n_colloc is None:
            self.n_colloc = self.t_train.shape[0]
        self.t_colloc,self.w_colloc = get_collocation_points_weights(self.t_train,self.n_colloc)


    def generate_train_data(
            self,   
            system,
            system_args, 
            system_solver,
            initial_state,
            rkey:jax.Array
    ):
        self.system_sol = system_solver(
            initial_state=initial_state,
            t0=self.t0,
            t1=self.t1,
            dt=self.ode_dt0,
            *system_args
        )
        self.x_true = jax.vmap(self.system_sol.evaluate)(self.t_true)
        self.x_train_true = jax.vmap(self.system_sol.evaluate)(self.t_train)
        self.x_train = self.x_train_true + self.noise*jax.random.normal(rkey,self.x_train_true.shape)
        self.x_dot = jnp.array([system(None,xi,system_args) for xi in self.x_true])

    def generate_test_data(
            self, 
            system,  
            system_solver,
            system_args, 
            initial_state,
            rkey:jax.Array
    ):
        self.test_initial_state = (
            initial_state
            +self.ic_std*jax.random.normal(rkey,initial_state.shape[0])
        )
        self.test_system_sol = system_solver(
            initial_state=self.test_initial_state,
            t0=self.t0,
            t1=self.t1,
            dt=self.dt,
            *system_args,
        )
        self.x_test = jax.vmap(self.test_system_sol.evaluate)(self.t_true)
        self.x_dot_test = jnp.array([system(None,xi,system_args) for xi in self.x_test])

@dataclass
class LinearExp(ExpData):
    A1 = jnp.array([[0, 1],
                    [-2, -3]])
    A2 = jnp.array([[0, -1],
                    [4,  0]])
    A3 = jnp.array([[-1, 2],
                    [-2, -1]])
    initial_state: jax.Array = field(
        default_factory=lambda: jnp.array([1., 0., 0., 1., -1., 0.])
    )
    feature_names = None

    def __post_init__(self):
        super().__post_init__()

        self.args = (self.A1,self.A2,self.A3)
        self.true_coeff = self.generate_true_coeff()

        # self.x_true = jax.vmap(self.system_sol.evaluate)(self.t_true)

        self.generate_train_data(
            system=linear_system,
            system_args = self.args,
            system_solver=solve_linear,
            initial_state=self.initial_state,
            rkey=self.train_key,
        )

        self.generate_test_data(
            system=linear_system,
            system_solver=solve_linear,
            system_args=self.args,
            initial_state=self.initial_state,
            rkey=self.test_key
        )
    def generate_true_coeff(self):
        # if poly lib with deg = 2
        theta = np.zeros((6, 28))
        theta = np.zeros((6, 7))
        theta[:2,1:3]  = self.A1
        theta[2:4,3:5] = self.A2
        theta[4:6,5:7] = self.A3

        return jnp.array(theta)

    def equations(self, coef, precision:int=3):
        sys_coord_names = self.feature_names
        feat_lib = ps.PolynomialLibrary()
        feat_lib.fit(self.x_train)
        feat_names = feat_lib.get_feature_names(sys_coord_names)

        def term(c, name):
            rounded_coef = jnp.round(c, precision)
            if rounded_coef == 0:
                return ""
            else:
                return f"{c:.{precision}f} {name}"

        equations = []
        for coef_row in coef:
            components = [term(c, i) for c, i in zip(coef_row, feat_names)]
            eq = " + ".join(filter(bool, components))
            if not eq:
                eq = f"{0:.{precision}f}"
            equations.append(eq)

        return equations

    def print(self, precision: int = 3, **kwargs) -> None:
        """Print the SINDy model equations.
        precision: int, optional (default 3)
            Precision to be used when printing out model coefficients.
        **kwargs: Additional keyword arguments passed to the builtin print function
        """
        eqns = self.equations(coef = self.true_coeff,precision = precision)
        if self.feature_names is None:
            feature_names = [f"x{i}" for i in range(len(eqns))]
        else:
            feature_names = self.feature_names

        for name, eqn in zip(feature_names, eqns, strict=True):
            lhs = f"({name})'"
            print(f"{lhs} = {eqn}", **kwargs) 


@dataclass
class LorenzExp(ExpData):
    """
    Experment object for Lorenz system 

    Parameters
    ----------
    sigma, beta, rho: 
        params in lorenz system
    
    initial_state: 
        Initial system state
    
    """
    sigma:float = 10.0
    rho: float = 28.0
    beta: float = 8.0/3.0
    initial_state: jax.Array = field(default_factory=lambda: jnp.array([1.,1.,1.]))
    feature_names: Optional[list[str]] = None


    def __post_init__(self):
        super().__post_init__()

        self.lorenz_args = (self.sigma,self.rho,self.beta)
        self.true_coeff = self.generate_true_coeff()

        # self.x_true = jax.vmap(self.system_sol.evaluate)(self.t_true)

        self.generate_train_data(
            system=lorenz_system,
            system_args = self.lorenz_args,
            system_solver=solve_lorenz,
            initial_state=self.initial_state,
            rkey=self.train_key,
        )

        self.generate_test_data(
            system=lorenz_system,
            system_solver=solve_lorenz,
            system_args=self.lorenz_args,
            initial_state=self.initial_state,
            rkey=self.test_key
        )

    
    def generate_true_coeff(self):
        # if poly lib with deg = 2
        true_theta = np.zeros((3,10))
        true_theta[0,1] = -self.sigma
        true_theta[0,2]= self.sigma
        true_theta[1,1]=self.rho
        true_theta[1,2] = -1
        true_theta[1,6] = -1
        true_theta[2,5] = 1
        true_theta[2,3] = -self.beta

        return jnp.array(true_theta)

    def equations(self, coef, precision:int=3):
        sys_coord_names = self.feature_names
        feat_lib = ps.PolynomialLibrary()
        feat_lib.fit(self.x_train)
        feat_names = feat_lib.get_feature_names(sys_coord_names)

        def term(c, name):
            rounded_coef = jnp.round(c, precision)
            if rounded_coef == 0:
                return ""
            else:
                return f"{c:.{precision}f} {name}"

        equations = []
        for coef_row in coef:
            components = [term(c, i) for c, i in zip(coef_row, feat_names)]
            eq = " + ".join(filter(bool, components))
            if not eq:
                eq = f"{0:.{precision}f}"
            equations.append(eq)

        return equations

    def print(self, precision: int = 3, **kwargs) -> None:
        """Print the SINDy model equations.
        precision: int, optional (default 3)
            Precision to be used when printing out model coefficients.
        **kwargs: Additional keyword arguments passed to the builtin print function
        """
        eqns = self.equations(coef = self.true_coeff,precision = precision)
        if self.feature_names is None:
            feature_names = [f"x{i}" for i in range(len(eqns))]
        else:
            feature_names = self.feature_names

        for name, eqn in zip(feature_names, eqns, strict=True):
            lhs = f"({name})'"
            print(f"{lhs} = {eqn}", **kwargs)

    

@dataclass
class LotkaVolterraExp(ExpData):
    alpha: float = 1.1
    beta: float = 0.4
    gamma: float = 0.4
    delta: float = 0.1
    initial_state: jax.Array = field(default_factory=lambda: jnp.array([10.,5.]))
    feature_names: Optional[list[str]] = None

    def __post_init__(self):
        super().__post_init__()

        self.lv_args = (self.alpha,self.beta,self.gamma, self.delta)
        self.true_coeff = self.generate_true_coeff()

        self.generate_train_data(
            system=lotka_volterra_system,
            system_args = self.lv_args,
            system_solver=solve_lotka_voltera,
            initial_state=self.initial_state,
            rkey=self.train_key
        )

        self.generate_test_data(
            system=lotka_volterra_system,
            system_solver=solve_lotka_voltera,
            system_args=self.lv_args,
            initial_state=self.initial_state,
            rkey=self.test_key
        )

        self.x_true = jax.vmap(self.system_sol.evaluate)(self.t_true)
    
    def generate_true_coeff(self):
        true_theta = np.zeros((2,6))
        true_theta[0,1] = self.alpha
        true_theta[0,4] = -self.beta
        true_theta[1,2] = -self.gamma
        true_theta[1,4] = self.delta

        return jnp.array(true_theta)

    def equations(self, coef, precision:int=3):
        sys_coord_names = self.feature_names
        feat_lib = ps.PolynomialLibrary()
        feat_lib.fit(self.x_train)
        feat_names = feat_lib.get_feature_names(sys_coord_names)

        def term(c, name):
            rounded_coef = jnp.round(c, precision)
            if rounded_coef == 0:
                return ""
            else:
                return f"{c:.{precision}f} {name}"

        equations = []
        for coef_row in coef:
            components = [term(c, i) for c, i in zip(coef_row, feat_names)]
            eq = " + ".join(filter(bool, components))
            if not eq:
                eq = f"{0:.{precision}f}"
            equations.append(eq)

        return equations

    def print(self, precision: int = 3, **kwargs) -> None:
        """Print the SINDy model equations.
        precision: int, optional (default 3)
            Precision to be used when printing out model coefficients.
        **kwargs: Additional keyword arguments passed to the builtin print function
        """
        eqns = self.equations(coef = self.true_coeff,precision = precision)
        if self.feature_names is None:
            feature_names = [f"x{i}" for i in range(len(eqns))]
        else:
            feature_names = self.feature_names

        for name, eqn in zip(feature_names, eqns, strict=True):
            lhs = f"({name})'"
            print(f"{lhs} = {eqn}", **kwargs)

@dataclass
class RosslerExp(ExpData):
    """
    Experment object for Rossler system 

    Parameters
    ----------
    a, b, c: 
        params in rossler system
    
    initial_state: 
        Initial system state
    
    """
    a:float = 0.2
    b: float = 0.2
    c: float = 5.7
    initial_state: jax.Array = field(default_factory=lambda: jnp.array([-6.,5.,0.]))
    feature_names: Optional[list[str]] = None

    def __post_init__(self):
        super().__post_init__()

        self.rossler_args = (self.a,self.b,self.c)
        self.true_coeff = self.generate_true_coeff()

        self.generate_train_data(
            system=rossler_system,
            system_args = self.rossler_args,
            system_solver=solve_rossler,
            initial_state=self.initial_state,
            rkey=self.train_key,
        )

        self.generate_test_data(
            system=rossler_system,
            system_solver=solve_rossler,
            system_args=self.rossler_args,
            initial_state=self.initial_state,
            rkey=self.test_key
        )

    
    def generate_true_coeff(self):
        # if poly lib with deg = 2
        true_theta = np.zeros((3,10))
        true_theta[0,2] = -1
        true_theta[0,3] = -1
        true_theta[1,1] = 1
        true_theta[1,2] = self.a
        true_theta[2,0] = self.b
        true_theta[2,3] = -self.c
        true_theta[2,6] = 1

        return jnp.array(true_theta)

    def equations(self, coef, precision:int=3):
        sys_coord_names = self.feature_names
        feat_lib = ps.PolynomialLibrary()
        feat_lib.fit(self.x_train)
        feat_names = feat_lib.get_feature_names(sys_coord_names)

        def term(c, name):
            rounded_coef = jnp.round(c, precision)
            if rounded_coef == 0:
                return ""
            else:
                return f"{c:.{precision}f} {name}"

        equations = []
        for coef_row in coef:
            components = [term(c, i) for c, i in zip(coef_row, feat_names)]
            eq = " + ".join(filter(bool, components))
            if not eq:
                eq = f"{0:.{precision}f}"
            equations.append(eq)

        return equations

    def print(self, precision: int = 3, **kwargs) -> None:
        """Print the SINDy model equations.
        precision: int, optional (default 3)
            Precision to be used when printing out model coefficients.
        **kwargs: Additional keyword arguments passed to the builtin print function
        """
        eqns = self.equations(coef = self.true_coeff,precision = precision)
        if self.feature_names is None:
            feature_names = [f"x{i}" for i in range(len(eqns))]
        else:
            feature_names = self.feature_names

        for name, eqn in zip(feature_names, eqns, strict=True):
            lhs = f"({name})'"
            print(f"{lhs} = {eqn}", **kwargs)